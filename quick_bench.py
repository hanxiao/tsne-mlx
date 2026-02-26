"""Quick benchmark of MLX vs sklearn t-SNE."""

import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as SklearnTSNE
from tsne_mlx import TSNE

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_test_data(n_samples):
    """Load test data."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    if n_samples > len(X):
        n_repeats = (n_samples // len(X)) + 1
        X = np.tile(X, (n_repeats, 1))[:n_samples]
        y = np.tile(y, n_repeats)[:n_samples]
    else:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X.astype(np.float32), y


def benchmark_size(n_samples, n_iter=500):
    """Benchmark a specific size."""
    print(f"\nTesting {n_samples} samples, {n_iter} iterations...")
    X, y = load_test_data(n_samples)
    
    # MLX
    print("  MLX t-SNE...", end=" ", flush=True)
    mlx_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
                    n_iter=n_iter, random_state=RANDOM_STATE)
    start = time.time()
    Y_mlx = mlx_tsne.fit_transform(X)
    mlx_time = time.time() - start
    print(f"{mlx_time:.2f}s")
    
    # sklearn
    print("  sklearn t-SNE...", end=" ", flush=True)
    sklearn_tsne = SklearnTSNE(n_components=2, perplexity=30, learning_rate=200,
                               max_iter=n_iter, random_state=RANDOM_STATE, verbose=0)
    start = time.time()
    Y_sklearn = sklearn_tsne.fit_transform(X)
    sklearn_time = time.time() - start
    print(f"{sklearn_time:.2f}s")
    
    speedup = sklearn_time / mlx_time
    print(f"  Speedup: {speedup:.2f}x")
    
    return {
        'n_samples': n_samples,
        'mlx_time': mlx_time,
        'sklearn_time': sklearn_time,
        'speedup': speedup
    }


if __name__ == "__main__":
    print("Quick t-SNE Benchmark")
    print("=" * 60)
    
    # Reduced iterations for faster testing
    results = []
    results.append(benchmark_size(5000, n_iter=500))
    results.append(benchmark_size(10000, n_iter=500))
    
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"{'Samples':<10} {'MLX (s)':<12} {'sklearn (s)':<12} {'Speedup':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n_samples']:<10} {r['mlx_time']:<12.2f} {r['sklearn_time']:<12.2f} {r['speedup']:<10.2f}x")
    print(f"{'='*60}")
