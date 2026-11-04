"""Provided helper functions. You do not need to change anything in this folder.

Coordinate convention:
- Images: img[row, col] or img[y, x]
- Keypoints: stored as [x, y] where x=col, y=row
"""

from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw keypoints [x, y] of shape (q, 2) on the image (BGR colors)."""
    if len(img.shape) == 2:
        img = img[:, :, None].repeat(3, 2)
    if keypoints is None:
        raise ValueError("Error! Keypoints should not be None")
    for p in np.array(keypoints).tolist():
        cv2.circle(img, (int(round(p[0])), int(round(p[1]))), thickness, color, -1)
    return img


def _show(img_bgr: np.ndarray, title: str, width: float) -> None:
    h, w = img_bgr.shape[:2]
    fig, ax = plt.subplots(figsize=(width, width * h / w))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_keypoints(img: np.ndarray, keypoints: np.ndarray, title: str = "") -> None:
    """Show the detected keypoints on the image."""
    drawn = draw_keypoints(img.copy(), keypoints, color=(0, 0, 255), thickness=2)
    _show(drawn, f"{title}: {keypoints.shape[0]} keypoints", width=5)


def show_keypoints_pair(
    img1: np.ndarray, keypoints1: np.ndarray, img2: np.ndarray, keypoints2: np.ndarray, title1: str = "", title2: str = ""
) -> None:
    """Show two images with their keypoints side by side in one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 12 * img1.shape[0] / (img1.shape[1] + img2.shape[1])))
    for ax, img, kp, title in [(axes[0], img1, keypoints1, title1), (axes[1], img2, keypoints2, title2)]:
        ax.imshow(cv2.cvtColor(draw_keypoints(img.copy(), kp, color=(0, 0, 255), thickness=2), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{title}: {kp.shape[0]} keypoints", fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    plt.show()


def show_matches(
    img1: np.ndarray,
    keypoints1: np.ndarray,
    img2: np.ndarray,
    keypoints2: np.ndarray,
    matches: np.ndarray,
    title: str = "",
) -> None:
    """Show the matches (m, 2) between two images side by side, one line per match."""
    from matplotlib.patches import ConnectionPatch

    fig, axes = plt.subplots(1, 2, figsize=(12, 12 * img1.shape[0] / (img1.shape[1] + img2.shape[1])))
    for ax, img, kp in [(axes[0], img1, keypoints1), (axes[1], img2, keypoints2)]:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.scatter(kp[:, 0], kp[:, 1], s=8, c="red", linewidths=0)
        ax.axis("off")
    for i, j in matches:
        fig.add_artist(ConnectionPatch(
            xyA=keypoints1[i], coordsA=axes[0].transData, xyB=keypoints2[j], coordsB=axes[1].transData,
            color="blue", linewidth=0.5, alpha=0.7,
        ))
    fig.suptitle(f"{title}: {matches.shape[0]} matches", fontsize=10)
    fig.tight_layout()
    plt.show()
