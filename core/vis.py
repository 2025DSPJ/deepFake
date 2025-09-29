import os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def save_heatmap(per_frame_conf, out_path):
    if not per_frame_conf:
        return None
    arr = np.array(per_frame_conf, dtype=np.float32)[None, :]
    fig = plt.figure(figsize=(10, 2.0))
    ax = plt.subplot(111)
    im = ax.imshow(arr, aspect='auto', vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("Fake confidence (0–1)")
    ax.set_yticks([]); ax.set_xlabel("Frame index")
    ax.set_title("Per-frame Fake Confidence Heatmap")
    fig.tight_layout()
    _save_fig(fig, out_path)
    return out_path
