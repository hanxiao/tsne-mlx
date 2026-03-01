"""tsne-mlx Fashion-MNIST 70K animation. 750 iters, 120fps, real timestamps."""
import os, time
import numpy as np
import mlx.core as mx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.datasets import fetch_openml
from tsne_mlx.tsne import TSNE

FPS = 120
SEED = 42
N_ITER = 750

print("Loading Fashion-MNIST 70K...")
fm = fetch_openml('Fashion-MNIST', version=1, as_frame=False, parser='auto')
X = fm.data.astype(np.float32)
y = fm.target.astype(np.int32)
n = X.shape[0]

np.random.seed(SEED)
viz_idx = np.random.choice(n, 15000, replace=False)
y_viz = y[viz_idx]
colors = plt.cm.tab10(y_viz / 9.0)

snaps = []
snap_times = []
t_global = time.time()

def callback(epoch, Y_np):
    snaps.append(Y_np[viz_idx].copy())
    snap_times.append(time.time() - t_global)

print("Running tsne-mlx...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=N_ITER, random_state=SEED, verbose=250)
tsne.fit_transform(X, epoch_callback=callback)
t_total = time.time() - t_global
print(f"Done: {t_total:.2f}s, {len(snaps)} snapshots")

# Build animation
n_snap = len(snaps)

def get_square_lims(emb, margin=0.1):
    cx = (emb[:, 0].min() + emb[:, 0].max()) / 2
    cy = (emb[:, 1].min() + emb[:, 1].max()) / 2
    span = max(emb[:, 0].max() - emb[:, 0].min(), emb[:, 1].max() - emb[:, 1].min())
    hs = span / 2 * (1 + margin)
    return (cx - hs, cx + hs), (cy - hs, cy + hs)

xlim, ylim = get_square_lims(snaps[-1])

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
fig.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect('equal')
ax.axis('off')

scatter = ax.scatter([], [], s=1.5, alpha=0.6)
title = ax.set_title('', color='white', fontsize=14, pad=10, fontfamily='monospace')

init_f = 60     # 0.5s hold on init
hold_f = 240    # 2s hold on final
total_f = init_f + n_snap + hold_f

print(f"Rendering {total_f} frames at {FPS}fps...")

def update(frame):
    if frame < init_f:
        idx = 0
        t = snap_times[0]
        label = f'tsne-mlx  Fashion-MNIST  70,000 x 784  init  t={t:.2f}s'
    elif frame < init_f + n_snap:
        idx = frame - init_f
        t = snap_times[idx]
        label = f'tsne-mlx  Fashion-MNIST  70,000 x 784  iter {idx}/{N_ITER}  t={t:.2f}s'
    else:
        idx = n_snap - 1
        label = f'tsne-mlx  Fashion-MNIST  70,000 x 784  done in {t_total:.1f}s'

    scatter.set_offsets(snaps[idx])
    scatter.set_color(colors)
    title.set_text(label)
    return scatter, title

anim = animation.FuncAnimation(fig, update, frames=total_f, blit=True, interval=1000 // FPS)
outpath = '/Users/hanxiao/.openclaw/workspace/tsne-mlx/animation.mp4'
anim.save(outpath, writer=animation.FFMpegWriter(fps=FPS, bitrate=8000,
          extra_args=['-pix_fmt', 'yuv420p']))
plt.close()
print(f"Saved {outpath} ({os.path.getsize(outpath) / 1024 / 1024:.1f} MB)")
