"""Provided helper functions. You do not need to change anything in this folder.

What is the Harris detector invariant to? Apply a known change to an image, run YOUR detector
with fixed parameters on both, and check which keypoints are found again.
"""

from typing import Callable, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

TOL = 2.0  # a keypoint counts as re-detected if one is found within TOL pixels of its mapped position
MARGIN = 8  # keypoints mapped closer than this to the border of the changed image are ignored


def transform(img: np.ndarray, angle: float = 0.0, scale: float = 1.0, brightness: float = 0.0, gain: float = 1.0):
    """Rotate about the centre, resize, then I -> gain * I + brightness (clipped to [0, 255]).

    Returns the changed image and the 3x3 matrix H mapping pixel coordinates of `img` to it.
    """
    h, w = img.shape[:2]
    H = np.eye(3)
    out = img
    if angle != 0.0:
        H[:2] = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
        out = cv2.warpAffine(out, H[:2], (w, h), flags=cv2.INTER_LINEAR)
    if scale != 1.0:
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        sx, sy = new_w / w, new_h / h
        S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1]])
        out = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        H = S @ H
    if gain != 1.0 or brightness != 0.0:
        out = np.clip(out.astype(float) * gain + brightness, 0, 255).astype(np.uint8)
    return out, H


def map_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    p = np.concatenate([points.astype(float), np.ones((len(points), 1))], axis=1) @ H.T
    return p[:, :2] / p[:, 2:3]


class InvarianceExplorer:
    """Runs YOUR extract_harris on changed versions of an image and measures repeatability."""

    def __init__(self, img: np.ndarray, extract_harris: Callable, sigma: float, k: float, thresh: float):
        self.img, self.extract_harris = img, extract_harris
        self.sigma, self.k, self.thresh = sigma, k, thresh
        self.kp_ref, _ = extract_harris(img, sigma, k, thresh)

    def detect(self, img: np.ndarray, mode: str, sigma: float = None) -> np.ndarray:
        """mode 'absolute': fixed threshold T. mode 'top-K': the K strongest responses, K = number of keypoints in the original."""
        sigma = self.sigma if sigma is None else sigma
        if mode == "absolute":
            return self.extract_harris(img, sigma, self.k, self.thresh)[0]
        kp, C = self.extract_harris(img, sigma, self.k, 0.0)
        order = np.argsort(-C[kp[:, 1], kp[:, 0]])
        return kp[order[: len(self.kp_ref)]]

    def evaluate(self, mode: str = "absolute", sigma: float = None, **change) -> Dict:
        changed, H = transform(self.img, **change)
        kp = self.detect(changed, mode, sigma)
        mapped = map_points(H, self.kp_ref)
        h, w = changed.shape[:2]
        visible = (mapped[:, 0] >= MARGIN) & (mapped[:, 0] < w - MARGIN) & (mapped[:, 1] >= MARGIN) & (mapped[:, 1] < h - MARGIN)
        if visible.sum() == 0 or len(kp) == 0:
            found = np.zeros(len(mapped), bool)
        else:
            d = np.linalg.norm(mapped[:, None, :] - kp[None, :, :].astype(float), axis=2)
            found = visible & (d.min(axis=1) <= TOL)
        rep = found.sum() / max(visible.sum(), 1)
        return dict(changed=changed, keypoints=kp, mapped=mapped, visible=visible, found=found, repeatability=rep)

    def show(self, mode: str = "absolute", title: str = "", **change) -> None:
        r = self.evaluate(mode, **change)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        vis, fnd, m = r["visible"], r["found"], r["mapped"]
        ref = self.kp_ref
        axes[0].imshow(self.img, cmap="gray", vmin=0, vmax=255)
        axes[0].scatter(ref[vis, 0], ref[vis, 1], s=12, c="red")
        axes[0].scatter(ref[~vis, 0], ref[~vis, 1], s=12, c="gray", label="ignored (border)")
        axes[0].set_title(f"original: {len(ref)} keypoints, {vis.sum()} evaluated", fontsize=10)
        axes[0].legend(loc="lower right", fontsize=8, framealpha=0.8)
        axes[1].imshow(r["changed"], cmap="gray", vmin=0, vmax=255)
        axes[1].scatter(m[vis & fnd, 0], m[vis & fnd, 1], s=12, c="lime", label="re-detected")
        axes[1].scatter(m[vis & ~fnd, 0], m[vis & ~fnd, 1], s=12, c="red", label="lost")
        axes[1].scatter(m[~vis, 0], m[~vis, 1], s=12, c="gray", label="ignored (border)")
        new = r["keypoints"]
        axes[1].scatter(new[:, 0], new[:, 1], s=30, facecolors="none", edgecolors="cyan", linewidths=0.6, label="detected now")
        labels = {"angle": "rotate {:g}°", "scale": "resize {:g}", "brightness": "I + {:g}", "gain": "{:g} · I"}
        change_txt = ", ".join(labels[k].format(v) for k, v in change.items() if v not in (0, 0.0, 1, 1.0)) or "no change"
        axes[1].set_title(f"{title or change_txt} ({mode}): {len(new)} keypoints, repeatability {r['repeatability']:.2f}", fontsize=10)
        axes[1].legend(loc="lower right", fontsize=8, framealpha=0.8)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        plt.show()

    def interact(self) -> None:
        """Sliders for the four changes and a switch for the threshold mode. Needs ipywidgets."""
        from ipywidgets import FloatSlider, ToggleButtons, interact

        s = dict(continuous_update=False, layout={"width": "500px"}, style={"description_width": "90px"})
        interact(
            lambda angle, scale, brightness, gain, mode: self.show(mode, angle=angle, scale=scale, brightness=brightness, gain=gain),
            angle=FloatSlider(0, min=0, max=90, step=5, description="rotation [°]", **s),
            scale=FloatSlider(1, min=0.5, max=2, step=0.1, description="image resize", **s),
            brightness=FloatSlider(0, min=-100, max=100, step=10, description="I + c", **s),
            gain=FloatSlider(1, min=0.25, max=2, step=0.25, description="a · I", **s),
            mode=ToggleButtons(options=["absolute", "top-K"], description="threshold"),
        )

    def summary(self, presets: Dict[str, Tuple[Dict, str]] = None) -> None:
        """Bar chart of the repeatability for a few fixed changes, in both threshold modes."""
        presets = presets or {
            "I + 40": dict(brightness=40), "0.5 · I": dict(gain=0.5), "rotate 90°": dict(angle=90),
            "resize 0.5": dict(scale=0.5), "resize 2": dict(scale=2.0),
        }
        names = list(presets)
        fig, ax = plt.subplots(figsize=(10, 4))
        x = np.arange(len(names))
        for i, mode in enumerate(["absolute", "top-K"]):
            reps = [self.evaluate(mode, **presets[n])["repeatability"] for n in names]
            bars = ax.bar(x + (i - 0.5) * 0.38, reps, 0.38, label=f"{mode} threshold")
            for b, r in zip(bars, reps):
                ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
        counts = [len(self.evaluate("absolute", **presets[n])["keypoints"]) for n in names]
        ax.set_xticks(x, [f"{n}\n{c} keypoints with C > T" for n, c in zip(names, counts)], fontsize=9)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("repeatability")
        ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()
