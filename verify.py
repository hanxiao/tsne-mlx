"""Verify MLX t-SNE against sklearn and benchmark performance."""

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE as SklearnTSNE
from tsne_mlx import TSNE

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def load_test_data(n_samples=5000):
    """Load and prepare test data (MNIST digits subset)."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    
    # If we need more samples, replicate the dataset
    if n_samples > len(X):
        n_repeats = (n_samples // len(X)) + 1
        X = np.tile(X, (n_repeats, 1))[:n_samples]
        y = np.tile(y, n_repeats)[:n_samples]
    else:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X.astype(np.float32), y


def compare_methods(X, y, n_samples):
    """Compare MLX and sklearn implementations."""
    print(f"\n{'='*60}")
    print(f"Testing with {n_samples} samples")
    print(f"{'='*60}\n")
    
    # MLX t-SNE
    print("Running MLX t-SNE...")
    mlx_tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        n_iter=1000,
        random_state=RANDOM_STATE
    )
    start_time = time.time()
    Y_mlx = mlx_tsne.fit_transform(X)
    mlx_time = time.time() - start_time
    print(f"MLX t-SNE completed in {mlx_time:.2f} seconds\n")
    
    # Sklearn t-SNE
    print("Running sklearn t-SNE...")
    sklearn_tsne = SklearnTSNE(
        n_components=2,
        perplexity=30,
        learning_rate=200,
        max_iter=1000,
        random_state=RANDOM_STATE
    )
    start_time = time.time()
    Y_sklearn = sklearn_tsne.fit_transform(X)
    sklearn_time = time.time() - start_time
    print(f"sklearn t-SNE completed in {sklearn_time:.2f} seconds\n")
    
    # Report speedup
    speedup = sklearn_time / mlx_time
    print(f"Speedup: {speedup:.2f}x")
    
    return Y_mlx, Y_sklearn, mlx_time, sklearn_time


def visualize_comparison(Y_mlx, Y_sklearn, y, n_samples, filename="comparison.png"):
    """Create side-by-side visualization of both methods."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot MLX results
    scatter1 = ax1.scatter(Y_mlx[:, 0], Y_mlx[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    ax1.set_title(f'MLX t-SNE ({n_samples} samples)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Component 1')
    ax1.set_ylabel('Component 2')
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Digit')
    
    # Plot sklearn results
    scatter2 = ax2.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1], c=y, cmap='tab10', s=10, alpha=0.7)
    ax2.set_title(f'sklearn t-SNE ({n_samples} samples)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Component 1')
    ax2.set_ylabel('Component 2')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=ax2, label='Digit')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {filename}")
    plt.close()


def run_benchmarks():
    """Run benchmarks on different dataset sizes."""
    sizes = [5000, 10000]
    results = []
    
    for n_samples in sizes:
        print(f"\n{'#'*60}")
        print(f"# Benchmark: {n_samples} samples")
        print(f"{'#'*60}")
        
        X, y = load_test_data(n_samples)
        Y_mlx, Y_sklearn, mlx_time, sklearn_time = compare_methods(X, y, n_samples)
        
        results.append({
            'n_samples': n_samples,
            'mlx_time': mlx_time,
            'sklearn_time': sklearn_time,
            'speedup': sklearn_time / mlx_time
        })
        
        # Generate visualization for first benchmark
        if n_samples == 5000:
            visualize_comparison(Y_mlx, Y_sklearn, y, n_samples)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Samples':<10} {'MLX (s)':<12} {'sklearn (s)':<12} {'Speedup':<10}")
    print(f"{'-'*60}")
    for r in results:
        print(f"{r['n_samples']:<10} {r['mlx_time']:<12.2f} {r['sklearn_time']:<12.2f} {r['speedup']:<10.2f}x")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("t-SNE MLX Verification and Benchmark")
    print("=" * 60)
    run_benchmarks()
