# tsne-mlx

Pure MLX t-SNE implementation for Apple Silicon. Runs the exact t-SNE algorithm entirely on Metal GPU via MLX.

## Install

```bash
pip install tsne-mlx
```

Or from source:

```bash
git clone https://github.com/hanxiao/tsne-mlx.git
cd tsne-mlx
pip install -e .
```

## Usage

API mirrors scikit-learn:

```python
from tsne_mlx import TSNE
import numpy as np

X = np.random.randn(1000, 50).astype(np.float32)
Y = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
```

Parameters:

- `n_components`: output dimensions (default: 2)
- `perplexity`: effective number of neighbors (default: 30)
- `learning_rate`: gradient descent step size (default: 200)
- `n_iter`: number of optimization iterations (default: 1000)
- `early_exaggeration`: factor for early phase (default: 12)
- `random_state`: seed for reproducibility
- `verbose`: set to 1 for progress logging

## Benchmark

Tested on M3 Ultra (512GB), 1000 iterations, MNIST digits:

| Samples | MLX (s) | sklearn (s) |
|---------|---------|-------------|
| 1,000   | 0.68    | 0.42        |
| 5,000   | 8.87    | 5.60        |
| 10,000  | 34.59   | 8.89        |

sklearn uses Barnes-Hut approximation (O(n log n) per iteration) by default. This implementation uses the exact algorithm (O(n^2) per iteration), which is more accurate but scales quadratically. The gap widens with dataset size as expected.

For datasets under ~3K points, performance is comparable. The exact method produces tighter, more well-separated clusters.

## Comparison

Side-by-side with sklearn on 5K MNIST digits:

![comparison](comparison.png)

Both produce well-separated digit clusters. The MLX exact method tends to produce tighter, more compact clusters compared to sklearn's Barnes-Hut approximation.

## Implementation

- Pairwise distance computation on Metal GPU
- Vectorized binary search for perplexity-based Gaussian widths
- Symmetrized joint probabilities
- Gradient descent with adaptive gains, momentum scheduling, and early exaggeration
- Student t-distribution (df=1) for low-dimensional affinities

Dependencies: `mlx`, `numpy`. No PyTorch, no CUDA.

## License

Apache 2.0
