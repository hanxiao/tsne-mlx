"""t-SNE implementation in pure MLX."""

import mlx.core as mx
import numpy as np


def _pairwise_distances(X):
    """Compute pairwise squared Euclidean distances on GPU."""
    sum_X = mx.sum(X * X, axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2.0 * (X @ X.T)
    return mx.maximum(D, 0.0)


def _binary_search_perplexity_vectorized(D_np, target_perplexity, tol=1e-5, max_iter=100):
    """Vectorized binary search for perplexity across all points simultaneously.
    
    Operates in numpy for the binary search (one-time cost), returns MLX array.
    D_np: numpy array of pairwise distances (n, n).
    """
    n = D_np.shape[0]
    target_entropy = np.log(target_perplexity)

    # Initialize
    beta = np.ones(n)        # precision = 1/(2*sigma^2)
    beta_min = np.full(n, -np.inf)
    beta_max = np.full(n, np.inf)

    # Zero out diagonal
    np.fill_diagonal(D_np, np.inf)

    P = np.zeros((n, n))

    for _ in range(max_iter):
        # Compute P_j|i for all i simultaneously: (n, n)
        # P_j|i = exp(-beta_i * D_ij) / sum_j exp(-beta_i * D_ij)
        exponents = -D_np * beta[:, None]
        # Numerical stability: subtract max per row
        exponents -= exponents.max(axis=1, keepdims=True)
        Pi = np.exp(exponents)
        np.fill_diagonal(Pi, 0.0)
        sum_Pi = Pi.sum(axis=1)
        sum_Pi = np.maximum(sum_Pi, 1e-12)

        Pi_norm = Pi / sum_Pi[:, None]

        # Entropy: H_i = log(sum_Pi) + beta_i * sum_j(D_ij * P_j|i_unnorm) / sum_Pi
        # Simplified: H_i = -sum_j P_j|i * log(P_j|i)
        log_Pi = np.log(np.maximum(Pi_norm, 1e-12))
        H = -np.sum(Pi_norm * log_Pi, axis=1)

        H_diff = H - target_entropy

        # Check convergence
        converged = np.abs(H_diff) < tol
        if converged.all():
            break

        # Update beta: if H > target, need larger beta (narrower Gaussian)
        need_increase = H_diff > 0
        need_decrease = ~need_increase & ~converged

        # Increase beta
        mask = need_increase & ~converged
        beta_min[mask] = beta[mask]
        finite = mask & np.isfinite(beta_max)
        beta[finite] = (beta[finite] + beta_max[finite]) / 2.0
        infinite = mask & ~np.isfinite(beta_max)
        beta[infinite] = beta[infinite] * 2.0

        # Decrease beta
        mask = need_decrease
        beta_max[mask] = beta[mask]
        finite = mask & np.isfinite(beta_min)
        beta[finite] = (beta[finite] + beta_min[finite]) / 2.0
        infinite = mask & ~np.isfinite(beta_min)
        beta[infinite] = beta[infinite] / 2.0

    P = Pi_norm
    return P


class TSNE:
    """t-SNE dimensionality reduction using MLX.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.
    perplexity : float, default=30.0
        Related to the number of nearest neighbors. Consider values between 5 and 50.
    learning_rate : float or "auto", default=200.0
        Learning rate for optimization.
    n_iter : int, default=1000
        Maximum number of gradient descent iterations.
    early_exaggeration : float, default=12.0
        Controls tightness of natural clusters in the embedding.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level. 1 prints progress every 50 iterations.
    """

    def __init__(
        self,
        n_components=2,
        perplexity=30.0,
        learning_rate=200.0,
        n_iter=1000,
        early_exaggeration=12.0,
        random_state=None,
        verbose=0,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.random_state = random_state
        self.verbose = verbose
        self.embedding_ = None
        self.kl_divergence_ = None

    def fit_transform(self, X):
        """Fit t-SNE and return embedded coordinates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        Y : ndarray of shape (n_samples, n_components)
            Embedding coordinates (numpy array).
        """
        if isinstance(X, mx.array):
            X_np = np.array(X)
        elif isinstance(X, np.ndarray):
            X_np = X.astype(np.float32)
        else:
            X_np = np.array(X, dtype=np.float32)

        n = X_np.shape[0]

        # Step 1: Compute pairwise distances (use MLX for GPU acceleration)
        X_mx = mx.array(X_np)
        D_mx = _pairwise_distances(X_mx)
        mx.eval(D_mx)
        D_np = np.array(D_mx)

        if self.verbose:
            print("Computed pairwise distances")

        # Step 2: Binary search for perplexity (vectorized numpy)
        P_np = _binary_search_perplexity_vectorized(D_np, self.perplexity)

        # Symmetrize: P = (P + P^T) / (2n)
        P_np = (P_np + P_np.T) / (2.0 * n)
        P_np = np.maximum(P_np, 1e-12)

        if self.verbose:
            print("Computed joint probabilities")

        # Step 3: Gradient descent in MLX
        P = mx.array(P_np.astype(np.float32))

        # Initialize embedding
        if self.random_state is not None:
            np.random.seed(self.random_state)
        Y = mx.array(np.random.randn(n, self.n_components).astype(np.float32) * 1e-4)

        # Early exaggeration
        P_early = P * self.early_exaggeration
        early_exag_end = 250

        # Momentum schedule
        momentum_init = 0.5
        momentum_final = 0.8

        velocity = mx.zeros_like(Y)
        gains = mx.ones_like(Y)  # Adaptive learning rate
        lr = float(self.learning_rate)

        for it in range(self.n_iter):
            P_cur = P_early if it < early_exag_end else P
            mom = momentum_init if it < 20 else momentum_final

            # Compute Q (Student-t affinities in low-dim space)
            D_low = _pairwise_distances(Y)
            Q_num = 1.0 / (1.0 + D_low)
            # Zero diagonal
            Q_num = Q_num * (1.0 - mx.eye(n))
            Q_sum = mx.sum(Q_num)
            Q = mx.maximum(Q_num / Q_sum, 1e-12)

            # Gradient: 4 * sum_j (p_ij - q_ij)(y_i - y_j)(1+||y_i-y_j||^2)^{-1}
            PQ = P_cur - Q  # (n, n)
            # Weighted by inverse distances
            W = PQ * Q_num  # (n, n), element-wise

            # grad_i = 4 * sum_j W_ij * (y_i - y_j)
            # = 4 * (diag(sum_j W_ij) - W) @ Y
            # = 4 * (sum_rows * Y - W @ Y)
            W_rowsum = mx.sum(W, axis=1, keepdims=True)
            grad = 4.0 * (W_rowsum * Y - W @ Y)

            # Adaptive gains (like sklearn)
            grad_sign = grad > 0
            vel_sign = velocity > 0
            same_sign = (grad_sign == vel_sign)
            gains = mx.where(same_sign, gains * 0.8, gains + 0.2)
            gains = mx.maximum(gains, 0.01)

            velocity = mom * velocity - lr * gains * grad
            Y = Y + velocity

            # Center
            Y = Y - mx.mean(Y, axis=0, keepdims=True)

            mx.eval(Y, velocity, gains)

            if self.verbose and (it + 1) % 50 == 0:
                kl = float(mx.sum(P_cur * mx.log(P_cur / Q)))
                print(f"  Iteration {it + 1}: KL divergence = {kl:.4f}")

        self.embedding_ = Y
        self.kl_divergence_ = float(mx.sum(P * mx.log(P / Q)))

        return np.array(Y)

    def fit(self, X):
        """Fit t-SNE model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.fit_transform(X)
        return self
