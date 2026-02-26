"""Quick correctness test on small data."""
import time
import numpy as np
from sklearn.datasets import load_digits
from tsne_mlx import TSNE

np.random.seed(42)
digits = load_digits()
X = digits.data[:1000].astype(np.float32)
y = digits.target[:1000]

print(f"Testing on {len(X)} samples, {X.shape[1]} dims")

tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42, verbose=1)
start = time.time()
Y = tsne.fit_transform(X)
elapsed = time.time() - start

print(f"\nDone in {elapsed:.2f}s")
print(f"Output shape: {Y.shape}")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")
print(f"Y range: [{Y.min():.2f}, {Y.max():.2f}]")
print(f"Y std: {Y.std():.2f}")
