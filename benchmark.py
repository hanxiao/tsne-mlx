"""Benchmark MLX vs sklearn t-SNE and generate comparison plot."""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as SklearnTSNE
from tsne_mlx import TSNE

SEED = 42
np.random.seed(SEED)


def get_data(n):
    digits = load_digits()
    X, y = digits.data, digits.target
    if n > len(X):
        reps = (n // len(X)) + 1
        X = np.tile(X, (reps, 1))[:n]
        y = np.tile(y, reps)[:n]
    else:
        idx = np.random.choice(len(X), n, replace=False)
        X, y = X[idx], y[idx]
    return X.astype(np.float32), y


def bench(n, n_iter=1000):
    print(f"\n--- {n} samples, {n_iter} iterations ---")
    X, y = get_data(n)

    # MLX
    print("  MLX...", end=" ", flush=True)
    t0 = time.time()
    Y_mlx = TSNE(n_components=2, perplexity=30, n_iter=n_iter, random_state=SEED, verbose=0).fit_transform(X)
    mlx_t = time.time() - t0
    print(f"{mlx_t:.2f}s")

    # sklearn
    print("  sklearn...", end=" ", flush=True)
    t0 = time.time()
    Y_sk = SklearnTSNE(n_components=2, perplexity=30, max_iter=n_iter, random_state=SEED).fit_transform(X)
    sk_t = time.time() - t0
    print(f"{sk_t:.2f}s")

    ratio = sk_t / mlx_t
    print(f"  Ratio: {ratio:.2f}x ({'faster' if ratio > 1 else 'slower'})")
    return Y_mlx, Y_sk, y, mlx_t, sk_t


# Run 5K comparison and generate plot
Y_mlx, Y_sk, y, mlx_5k, sk_5k = bench(5000, n_iter=1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
for ax, Y, label, t in [(ax1, Y_mlx, "MLX", mlx_5k), (ax2, Y_sk, "sklearn", sk_5k)]:
    sc = ax.scatter(Y[:, 0], Y[:, 1], c=y, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"{label} t-SNE (5K points, {t:.1f}s)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax, label="Digit")
plt.tight_layout()
plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
print("\nSaved comparison.png")

# Run 10K benchmark
_, _, _, mlx_10k, sk_10k = bench(10000, n_iter=1000)

# Summary
print(f"\n{'='*60}")
print("BENCHMARK SUMMARY (M3 Ultra, 1000 iterations)")
print(f"{'='*60}")
print(f"{'Samples':<10} {'MLX (s)':<12} {'sklearn (s)':<14} {'Ratio':<10}")
print(f"{'-'*60}")
print(f"{'5,000':<10} {mlx_5k:<12.2f} {sk_5k:<14.2f} {sk_5k/mlx_5k:<10.2f}x")
print(f"{'10,000':<10} {mlx_10k:<12.2f} {sk_10k:<14.2f} {sk_10k/mlx_10k:<10.2f}x")
print(f"{'='*60}")
