"""Provided helper functions. You do not need to change anything in this folder.

What is patch matching invariant to? Transform image 2 in a known way, detect and match with YOUR
functions, and check which matches are correct (the transformation tells us where every keypoint of
image 1 should land).
"""

from typing import Callable, Dict

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import ConnectionPatch

from .invariance import map_points, transform

TOL = 3.0  # a match is correct if the matched keypoint is within TOL pixels of the mapped position
PATCH_SIZE = 9


class MatchInvarianceExplorer:
    """Matches image 1 against changed versions of image 2 and measures how many matches are correct."""

    def __init__(self, img1, img2, H12, extract_harris, filter_keypoints, extract_patches, ssd, matchers: Dict[str, Callable],
                 sigma: float, k: float, thresh: float):
        self.img1, self.img2, self.H12 = img1, img2, H12
        self.extract_harris, self.filter_keypoints, self.extract_patches, self.ssd = extract_harris, filter_keypoints, extract_patches, ssd
        self.matchers = matchers
        self.sigma, self.k, self.thresh = sigma, k, thresh
        self.kp1, self.desc1 = self.features(img1)

    def features(self, img):
        kp, _ = self.extract_harris(img, self.sigma, self.k, self.thresh)
        kp = self.filter_keypoints(img, kp, PATCH_SIZE)
        if len(kp) == 0:
            return kp, np.empty((0, PATCH_SIZE * PATCH_SIZE))
        return kp, self.extract_patches(img, kp, PATCH_SIZE)

    def evaluate(self, method: str, **change) -> Dict:
        changed, H2 = transform(self.img2, **change)
        kp2, desc2 = self.features(changed)
        matches = self.matchers[method](self.ssd(self.desc1, desc2)) if len(kp2) > 1 else np.empty((0, 2), int)
        should = map_points(H2 @ self.H12, self.kp1)  # where every keypoint of image 1 should be in the changed image 2
        if len(matches):
            correct = np.linalg.norm(should[matches[:, 0]] - kp2[matches[:, 1]], axis=1) <= TOL
        else:
            correct = np.zeros(0, bool)
        return dict(changed=changed, kp2=kp2, matches=matches, correct=correct)

    def show(self, method: str = "ratio", **change) -> None:
        r = self.evaluate(method, **change)
        img1, changed, kp2, matches, correct = self.img1, r["changed"], r["kp2"], r["matches"], r["correct"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 12 * max(img1.shape[0], changed.shape[0]) / (img1.shape[1] + changed.shape[1])))
        for ax, img, kp in [(axes[0], img1, self.kp1), (axes[1], changed, kp2)]:
            ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            ax.scatter(kp[:, 0], kp[:, 1], s=8, c="red", linewidths=0)
            ax.axis("off")
        for (i, j), ok in zip(matches, correct):
            fig.add_artist(ConnectionPatch(xyA=self.kp1[i], coordsA=axes[0].transData, xyB=kp2[j], coordsB=axes[1].transData,
                                           color="lime" if ok else "red", linewidth=0.6, alpha=0.8))
        labels = {"angle": "rotate {:g}°", "scale": "resize {:g}", "brightness": "I + {:g}", "gain": "{:g} · I"}
        change_txt = ", ".join(labels[k].format(v) for k, v in change.items() if v not in (0, 0.0, 1, 1.0)) or "no change"
        precision = correct.mean() if len(correct) else 0.0
        fig.suptitle(f"{change_txt}, {method}: {len(matches)} matches, {correct.sum()} correct (green), precision {precision:.2f}", fontsize=10)
        fig.tight_layout()
        plt.show()

    def interact(self) -> None:
        """Sliders for the four changes of image 2 and a switch for the matcher. Needs ipywidgets."""
        from ipywidgets import FloatSlider, ToggleButtons, interact

        s = dict(continuous_update=False, layout={"width": "500px"}, style={"description_width": "90px"})
        interact(
            lambda angle, scale, brightness, gain, method: self.show(method, angle=angle, scale=scale, brightness=brightness, gain=gain),
            angle=FloatSlider(0, min=0, max=90, step=5, description="rotation [°]", **s),
            scale=FloatSlider(1, min=0.5, max=2, step=0.1, description="image resize", **s),
            brightness=FloatSlider(0, min=-100, max=100, step=10, description="I + c", **s),
            gain=FloatSlider(1, min=0.25, max=2, step=0.25, description="a · I", **s),
            method=ToggleButtons(options=list(self.matchers), description="matcher"),
        )

    def summary(self, presets: Dict[str, Dict] = None) -> None:
        """Number of correct matches for a few fixed changes, for every matcher."""
        presets = presets or {
            "no change": {}, "I + 40": dict(brightness=40), "0.75 · I": dict(gain=0.75), "rotate 10°": dict(angle=10),
            "rotate 90°": dict(angle=90), "resize 0.8": dict(scale=0.8), "resize 1.5": dict(scale=1.5),
        }
        names = list(presets)
        fig, ax = plt.subplots(figsize=(11, 4))
        x = np.arange(len(names))
        n = len(self.matchers)
        for i, method in enumerate(self.matchers):
            results = [self.evaluate(method, **presets[name]) for name in names]
            total = [len(r["matches"]) for r in results]
            good = [int(r["correct"].sum()) for r in results]
            pos = x + (i - (n - 1) / 2) * 0.8 / n
            ax.bar(pos, total, 0.8 / n, color="lightgray")
            bars = ax.bar(pos, good, 0.8 / n, label=method)
            for b, g, t in zip(bars, good, total):
                ax.text(b.get_x() + b.get_width() / 2, t + 4, f"{g}/{t}", ha="center", fontsize=7)
        counts = [len(self.evaluate("one-way", **presets[n])["kp2"]) for n in names]
        ax.set_xticks(x, [f"{n}\n({c} keypoints)" for n, c in zip(names, counts)], fontsize=9)
        ax.set_ylabel("matches (colour = correct, grey = all)")
        ax.legend(frameon=False, fontsize=9, ncol=n, loc="upper center")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        plt.show()
