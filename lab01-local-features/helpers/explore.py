"""Provided helper functions. You do not need to change anything in this folder.

Interactive look at the local auto-correlation matrix M (hover and click, see PatchExplorer.hunt): what does the shift error E(u, v)
look like around a pixel, and what do the eigenvalues of M say about it?
"""

from typing import Callable, Dict, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

RADIUS = 4  # shifts u, v in [-RADIUS, RADIUS]
VIEW = 14  # half size of the zoomed patch
E_FLOOR = 0.02  # minimal height of the z axis, so that flat regions look flat


def _shift_error(I: np.ndarray, x: int, y: int, sigma: float) -> np.ndarray:
    """True E(u, v) = sum_w w [I(x+u, y+v) - I(x, y)]^2 with the same Gaussian window as M."""
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    half = ksize // 2
    margin = RADIUS + half + 1
    padded = cv2.copyMakeBorder(I, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
    cx, cy = x + margin, y + margin
    g = cv2.getGaussianKernel(ksize, sigma)
    weights = g @ g.T
    window = padded[cy - half : cy + half + 1, cx - half : cx + half + 1]
    E = np.zeros((2 * RADIUS + 1, 2 * RADIUS + 1))
    for v in range(-RADIUS, RADIUS + 1):
        for u in range(-RADIUS, RADIUS + 1):
            shifted = padded[cy + v - half : cy + v + half + 1, cx + u - half : cx + u + half + 1]
            E[v + RADIUS, u + RADIUS] = np.sum(weights * (shifted - window) ** 2)
    return E


class PatchExplorer:
    """Precomputes M for an image with YOUR functions and draws the analysis of single pixels."""

    def __init__(self, img: np.ndarray, image_gradients: Callable, auto_correlation_matrix: Callable, sigma: float):
        self.img = img
        self.I = img.astype(float) / 255.0
        self.sigma = sigma
        Ix, Iy = image_gradients(self.I)
        self.Mxx, self.Mxy, self.Myy = auto_correlation_matrix(Ix, Iy, sigma)

    def M(self, x: int, y: int) -> np.ndarray:
        return np.array([[self.Mxx[y, x], self.Mxy[y, x]], [self.Mxy[y, x], self.Myy[y, x]]])

    def show(self, x: int, y: int, name: str = "", correct: bool = None) -> None:
        """Image, zoomed patch with the eigenvectors of M, and the true E(u, v) with its quadratic approximation."""
        h, w = self.img.shape
        x, y = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        M = self.M(x, y)
        eigvals, eigvecs = np.linalg.eigh(M)  # ascending
        shifts = np.arange(-RADIUS, RADIUS + 1)
        U, V = np.meshgrid(shifts, shifts)
        E_true = _shift_error(self.I, x, y, self.sigma)
        E_quad = M[0, 0] * U**2 + 2 * M[0, 1] * U * V + M[1, 1] * V**2
        zmax = max(1.3 * E_true.max(), E_FLOOR)

        fig = plt.figure(figsize=(15, 4.6))
        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(self.img, cmap="gray")
        ax.plot(x, y, "+", color="red", ms=14, mew=2)
        if correct is None:
            ax.set_title(f"{name}  (x, y) = ({x}, {y})".strip(), fontsize=10)
        else:
            ax.set_title(f"{'✔' if correct else '✘'} {name}  (x, y) = ({x}, {y})", fontsize=11,
                         color="tab:green" if correct else "tab:red", fontweight="bold")
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        ax.set_xlim(x - VIEW, x + VIEW)
        ax.set_ylim(y + VIEW, y - VIEW)
        ax.add_patch(plt.Circle((x, y), 2 * self.sigma, fill=False, color="yellow", lw=1))
        scale = 0.9 * VIEW / np.sqrt(max(eigvals.max(), 1e-3))
        for lam, vec, color in zip(eigvals[::-1], eigvecs.T[::-1], ["tab:red", "tab:blue"]):
            d = scale * np.sqrt(max(lam, 0)) * vec
            ax.plot([x - d[0], x + d[0]], [y - d[1], y + d[1]], color=color, lw=2.5)
        ax.set_title(f"λ₁ = {eigvals[1]:.4f} (red)   λ₂ = {eigvals[0]:.4f} (blue)", fontsize=10)
        ax.axis("off")

        ax = fig.add_subplot(1, 3, 3, projection="3d")
        ax.plot_surface(U, V, E_true, cmap="viridis", vmin=0, vmax=zmax, linewidth=0, alpha=0.9)
        wire = np.where(E_quad <= zmax, E_quad, np.nan)  # the approximation leaves the box where it overshoots
        ax.plot_wireframe(U, V, wire, color="tab:red", linewidth=0.9)
        ax.set_zlim(0, zmax)
        ax.set_zticks(np.round(np.linspace(0, zmax, 4), 3))
        ax.set_xlabel("u")
        ax.set_ylabel("v")
        ax.set_title("surface: true E(u, v)   red grid: [u v] M [u v]ᵀ", fontsize=10)
        ax.view_init(elev=35, azim=-60)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ the hunt

    TARGETS = {
        "flat": "λ₁ < 0.0001",
        "vertical edge": "λ₁ ≫ λ₂, change along x",
        "horizontal edge": "λ₁ ≫ λ₂, change along y",
        "diagonal edge": "λ₁ ≫ λ₂, change at 25–65°",
        "corner": "λ₂ > 0.004",
        "texture": "both small, λ₂ > 0.3 λ₁",
    }

    def classify(self, x: int, y: int):
        """Which targets does the pixel (x, y) satisfy?"""
        lam2, lam1 = np.linalg.eigvalsh(self.M(x, y))
        M = self.M(x, y)
        angle = abs(np.degrees(0.5 * np.arctan2(2 * M[0, 1], M[0, 0] - M[1, 1])))  # 0 = along x, 90 = along y
        is_edge = lam1 > 5e-3 and lam2 < 0.05 * lam1
        hits = set()
        if lam1 < 1e-4:
            hits.add("flat")
        if is_edge and angle < 25:
            hits.add("vertical edge")
        if is_edge and angle > 65:
            hits.add("horizontal edge")
        if is_edge and 25 <= angle <= 65:
            hits.add("diagonal edge")
        if lam2 > 4e-3:
            hits.add("corner")
        if lam2 > 1e-4 and lam1 < 2e-3 and lam2 > 0.3 * lam1:
            hits.add("texture")
        return hits

    def hunt(self) -> None:
        """Move the mouse over the image, click to analyse a pixel. Needs `%matplotlib widget` (ipympl)."""
        import time

        h, w = self.img.shape
        self.found = getattr(self, "found", {})
        fig = plt.figure(figsize=(13.5, 4.6))
        for attr in ("header_visible", "footer_visible", "toolbar_visible"):
            if hasattr(fig.canvas, attr):
                setattr(fig.canvas, attr, False)
        grid = fig.add_gridspec(1, 4, width_ratios=[1.2, 1, 1.35, 0.95], wspace=0.35)
        ax_img, ax_zoom = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])
        ax_3d, ax_list = fig.add_subplot(grid[2], projection="3d"), fig.add_subplot(grid[3])

        ax_img.imshow(self.img, cmap="gray")
        ax_img.set_title("move the mouse, click to analyse", fontsize=10)
        ax_img.axis("off")
        (cross,) = ax_img.plot([], [], "+", color="red", ms=14, mew=2)
        (clicked,) = ax_img.plot([], [], "o", mfc="none", color="cyan", ms=9, mew=1.5)

        ax_zoom.imshow(self.img, cmap="gray", vmin=0, vmax=255)
        ax_zoom.axis("off")
        window = plt.Circle((0, 0), 2 * self.sigma, fill=False, color="yellow", lw=1)
        ax_zoom.add_patch(window)
        lines = [ax_zoom.plot([], [], color=c, lw=2.5)[0] for c in ("tab:red", "tab:blue")]

        ax_list.axis("off")
        ax_list.set_title("find a pixel that is ...", fontsize=10)
        items = {}
        for i, (name, hint) in enumerate(self.TARGETS.items()):
            items[name] = ax_list.text(0.02, 0.96 - 0.155 * i, "", fontsize=10, va="top", transform=ax_list.transAxes)
            ax_list.text(0.12, 0.905 - 0.155 * i, hint, fontsize=7.5, color="gray", va="top", transform=ax_list.transAxes)

        def refresh_list():
            for name, text in items.items():
                done = name in self.found
                text.set_text(("[x] " if done else "[  ] ") + name + (f"  {self.found[name]}" if done else ""))
                text.set_color("tab:green" if done else "black")

        def pixel(event):
            if event.inaxes is not ax_img or event.xdata is None:
                return None
            return int(np.clip(round(event.xdata), 0, w - 1)), int(np.clip(round(event.ydata), 0, h - 1))

        def update_zoom(x, y):
            eigvals, eigvecs = np.linalg.eigh(self.M(x, y))
            cross.set_data([x], [y])
            ax_zoom.set_xlim(x - VIEW, x + VIEW)
            ax_zoom.set_ylim(y + VIEW, y - VIEW)
            window.center = (x, y)
            for line, lam, vec in zip(lines, eigvals[::-1], eigvecs.T[::-1]):
                d = 0.9 * VIEW * np.sqrt(max(lam, 0) / 0.04) * vec  # same scale everywhere: 0.04 fills the patch
                line.set_data([x - d[0], x + d[0]], [y - d[1], y + d[1]])
            ax_zoom.set_title(f"({x}, {y})   λ₁ = {eigvals[1]:.4f}   λ₂ = {eigvals[0]:.4f}", fontsize=10)

        def update_surface(x, y):
            M = self.M(x, y)
            shifts = np.arange(-RADIUS, RADIUS + 1)
            U, V = np.meshgrid(shifts, shifts)
            E_true = _shift_error(self.I, x, y, self.sigma)
            E_quad = M[0, 0] * U**2 + 2 * M[0, 1] * U * V + M[1, 1] * V**2
            zmax = max(1.3 * E_true.max(), E_FLOOR)
            ax_3d.clear()
            ax_3d.plot_surface(U, V, E_true, cmap="viridis", vmin=0, vmax=zmax, linewidth=0, alpha=0.9)
            ax_3d.plot_wireframe(U, V, np.where(E_quad <= zmax, E_quad, np.nan), color="tab:red", linewidth=0.9)
            ax_3d.set_zlim(0, zmax)
            ax_3d.set_zticks(np.round(np.linspace(0, zmax, 4), 3))
            ax_3d.set_xlabel("u")
            ax_3d.set_ylabel("v")
            ax_3d.set_title("surface: true E(u, v)   red grid: [u v] M [u v]ᵀ", fontsize=9)
            ax_3d.tick_params(labelsize=7)

        last = [0.0]

        def on_move(event):
            p = pixel(event)
            if p is None or time.time() - last[0] < 0.04:
                return
            last[0] = time.time()
            update_zoom(*p)
            fig.canvas.draw_idle()

        def on_click(event):
            p = pixel(event)
            if p is None:
                return
            update_zoom(*p)
            update_surface(*p)
            clicked.set_data([p[0]], [p[1]])
            for name in self.classify(*p):
                self.found.setdefault(name, p)
            refresh_list()
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_move)
        fig.canvas.mpl_connect("button_press_event", on_click)
        self._hunt = (fig, on_move, on_click, ax_img)  # keep references alive
        fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.08)
        update_zoom(w // 2, h // 2)
        update_surface(w // 2, h // 2)
        refresh_list()
        plt.show()

    def show_answers(self, answers: Dict[str, Tuple[int, int]]) -> None:
        """Static figures for the pixels you found, each checked against its target, plus a summary line."""
        h, w = self.img.shape
        found = 0
        for name in self.TARGETS:
            if name not in answers:
                print(f"✘ {name}: missing in FOUND")
                continue
            x, y = answers[name]
            ok = 0 <= x < w and 0 <= y < h and name in self.classify(int(x), int(y))
            found += ok
            self.show(x, y, name, correct=ok)
        print(f"{found} / {len(self.TARGETS)} pixel types found correctly")
