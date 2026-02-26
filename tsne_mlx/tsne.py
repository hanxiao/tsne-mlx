"""t-SNE implementation in pure MLX for Apple Silicon."""

import mlx.core as mx
import numpy as np


class TSNE:
    """t-SNE dimensionality reduction using MLX on Metal GPU.

    Parameters:
        n_components: Dimension of the embedded space (default 2).
        perplexity: Related to the number of nearest neighbors (default 30.0).
        learning_rate: Learning rate for gradient descent (default 200.0).
        n_iter: Number of iterations (default 1000).
        early_exaggeration: Factor by which P is multiplied in early iterations (default 12.0).
        early_exaggeration_iter: Number of early exaggeration iterations (default 250).
        momentum_init: Initial momentum (default 0.5).
        momentum_final: Final momentum after early exaggeration (default 0.8).
        random_state: Random seed for reproducibility.
        verbose: Print progress every N iterations (0 = silent).
    """

    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        early_exaggeration: float = 12.0,
        early_exaggeration_iter: int = 250,
        momentum_init: float = 0.5,
        momentum_final: float = 0.8,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_exaggeration = early_exaggeration
        self.early_exaggeration_iter = early_exaggeration_iter
        self.momentum_init = momentum_init
        self.momentum_final = momentum_final
        self.random_state = random_state
        self.verbose = verbose
        self.embedding_ = None

    def fit_transform(self, X) -> np.ndarray:
        """Fit t-SNE and return the embedding.

        Args:
            X: Input data, shape (n_samples, n_features). Can be np.ndarray or mx.array.

        Returns:
            Embedding as np.ndarray, shape (n_samples, n_components).
        """
        if isinstance(X, np.ndarray):
            X = mx.array(X, dtype=mx.float32)
        elif X.dtype != mx.float32:
            X = X.astype(mx.float32)

        n = X.shape[0]

        # Step 1: Compute pairwise distances
        D = self._pairwise_distances(X)
        mx.eval(D)

        # Step 2: Compute joint probabilities P
        P = self._compute_joint_probabilities(D)
        mx.eval(P)

        # Free distance matrix
        del D

        # Step 3: Initialize embedding
        if self.random_state is not None:
            mx.random.seed(self.random_state)
        Y = mx.random.normal((n, self.n_components)) * 1e-4
        mx.eval(Y)

        # Step 4: Optimize with gradient descent
        Y = self._optimize(P, Y)
        mx.eval(Y)

        self.embedding_ = np.array(Y)
        return self.embedding_

    def _pairwise_distances(self, X: mx.array) -> mx.array:
        """Compute pairwise squared Euclidean distances. GPU-accelerated."""
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x.y
        sum_sq = mx.sum(X * X, axis=1)
        D = sum_sq[:, None] + sum_sq[None, :] - 2.0 * (X @ X.T)
        # Ensure non-negative (numerical precision)
        D = mx.maximum(D, 0.0)
        return D

    def _compute_joint_probabilities(self, D: mx.array) -> mx.array:
        """Compute symmetric joint probabilities from distances.

        Uses binary search to find per-point sigma that achieves target perplexity.
        """
        n = D.shape[0]
        target_entropy = mx.log(mx.array(self.perplexity))

        # Binary search for sigma per point (done in numpy for control flow)
        D_np = np.array(D)
        P = np.zeros((n, n), dtype=np.float32)

        for i in range(n):
            lo, hi = 1e-20, 1e5
            beta = 1.0  # 1 / (2 * sigma^2)

            for _ in range(50):  # binary search iterations
                # Compute conditional probabilities
                dists = D_np[i].copy()
                dists[i] = np.inf
                p_i = np.exp(-dists * beta)
                sum_p = max(p_i.sum(), 1e-20)
                p_i /= sum_p

                # Compute entropy
                entropy = -np.sum(p_i * np.log(np.maximum(p_i, 1e-20)))
                target = float(target_entropy.item())

                if abs(entropy - target) < 1e-5:
                    break

                if entropy > target:
                    lo = beta
                    beta = (beta + hi) / 2.0 if hi < 1e4 else beta * 2.0
                else:
                    hi = beta
                    beta = (beta + lo) / 2.0

                P[i] = p_i

            P[i] = p_i

        # Symmetrize and normalize
        P = (P + P.T) / (2.0 * n)
        P = np.maximum(P, 1e-12)

        return mx.array(P)

    def _optimize(self, P: mx.array, Y: mx.array) -> mx.array:
        """Gradient descent optimization of the KL divergence."""
        n = Y.shape[0]
        velocity = mx.zeros_like(Y)
        gains = mx.ones_like(Y)

        # Pre-compute constants (avoid recreating each iteration)
        eye_mask = 1.0 - mx.eye(n)
        P_exag = P * self.early_exaggeration

        for iteration in range(self.n_iter):
            P_curr = P_exag if iteration < self.early_exaggeration_iter else P
            momentum = self.momentum_init if iteration < self.early_exaggeration_iter else self.momentum_final

            # Compute gradient with cached eye mask
            grad = self._compute_gradient(P_curr, Y, eye_mask)

            # Adaptive gains
            same_sign = (mx.sign(grad) == mx.sign(velocity))
            gains = mx.where(same_sign, gains * 0.8, gains + 0.2)
            gains = mx.maximum(gains, 0.01)

            # Update
            velocity = momentum * velocity - self.learning_rate * gains * grad
            Y = Y + velocity
            Y = Y - mx.mean(Y, axis=0)

            # Eval once per iteration to keep graph small
            mx.eval(Y, velocity, gains)

            if self.verbose > 0 and (iteration + 1) % self.verbose == 0:
                Q = self._compute_q(Y, eye_mask)
                mx.eval(Q)
                kl = float(mx.sum(P_curr * mx.log(mx.maximum(P_curr, 1e-12) / mx.maximum(Q, 1e-12))).item())
                print(f"Iteration {iteration + 1}: KL divergence = {kl:.4f}")

        return Y

    def _compute_gradient(self, P: mx.array, Y: mx.array, eye_mask: mx.array) -> mx.array:
        """Compute t-SNE gradient using Student-t distribution.

        Avoids materializing (n, n, d) tensor by computing gradient per-dimension.
        Uses Python scalars (4.0, 1e-12) to avoid dtype upcasting.
        """
        # Pairwise squared distances
        sum_sq = mx.sum(Y * Y, axis=1)
        dist_sq = mx.maximum(sum_sq[:, None] + sum_sq[None, :] - 2.0 * (Y @ Y.T), 0.0)

        # Student-t kernel with pre-computed mask
        inv_dist = eye_mask / (1.0 + dist_sq)

        # Q distribution
        Q = mx.maximum(inv_dist / mx.sum(inv_dist), 1e-12)

        # Gradient: 4 * (diag(row_sums) @ Y - weights @ Y)
        weights = (P - Q) * inv_dist
        row_sums = mx.sum(weights, axis=1, keepdims=True)
        return 4.0 * (row_sums * Y - weights @ Y)

    def _compute_q(self, Y: mx.array, eye_mask: mx.array) -> mx.array:
        """Compute Q distribution for KL divergence logging."""
        sum_sq = mx.sum(Y * Y, axis=1)
        dist_sq = mx.maximum(sum_sq[:, None] + sum_sq[None, :] - 2.0 * (Y @ Y.T), 0.0)
        inv_dist = eye_mask / (1.0 + dist_sq)
        return mx.maximum(inv_dist / mx.sum(inv_dist), 1e-12)
