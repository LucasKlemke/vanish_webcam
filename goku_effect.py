"""Instant Transmission: two-finger forehead pose and comic speed-line teleport."""

from __future__ import annotations

import math

import cv2
import numpy as np


TELEPORT_SECONDS = 0.9
HOLD_SECONDS = 1.0

# Indici FaceMesh: alto della fronte, centro fronte, tra le sopracciglia, lati del viso.
FOREHEAD_INDICES = (10, 151, 9)
FACE_LEFT, FACE_RIGHT = 234, 454


def _pixels(landmarks, width: int, height: int) -> np.ndarray:
    return np.array([(p.x * width, p.y * height) for p in landmarks], dtype=np.float32)


def two_finger_hand(points: np.ndarray) -> bool:
    """Indice e medio distesi e uniti, anulare e mignolo piegati."""
    wrist = points[0]
    reach = [float(np.linalg.norm(points[i] - wrist)) for i in range(21)]
    scale = reach[9]
    if scale < 8.0:
        return False
    extended = all(reach[tip] > reach[mcp] * 1.45 for tip, mcp in ((8, 5), (12, 9)))
    folded = all(reach[tip] < reach[mcp] * 1.25 for tip, mcp in ((16, 13), (20, 17)))
    together = float(np.linalg.norm(points[8] - points[12])) < scale * 0.4
    return extended and folded and together


def goku_pose_from_landmarks(hands, face, width: int, height: int) -> bool:
    """Riconosci indice e medio uniti appoggiati sulla fronte."""
    if face is None or len(face) <= FACE_RIGHT:
        return False
    face_points = _pixels(face, width, height)
    face_width = float(np.linalg.norm(face_points[FACE_LEFT] - face_points[FACE_RIGHT]))
    if face_width < 20.0:
        return False
    forehead = face_points[list(FOREHEAD_INDICES)]
    for hand in hands:
        if len(hand) != 21:
            continue
        points = _pixels(hand, width, height)
        if not np.isfinite(points).all() or not two_finger_hand(points):
            continue
        tips = (points[8] + points[12]) * 0.5
        if float(np.min(np.linalg.norm(forehead - tips, axis=1))) < face_width * 0.3:
            return True
    return False


def _smoothstep(edge0: float, edge1: float, value: float) -> float:
    value = min(1.0, max(0.0, (value - edge0) / (edge1 - edge0)))
    return value * value * (3.0 - 2.0 * value)


def _speed_lines(output: np.ndarray, mask: np.ndarray, progress: float, strength: float) -> None:
    points = cv2.findNonZero((mask > 0.3).astype(np.uint8))
    height, width = output.shape[:2]
    if points is None:
        x, y, w, h = width // 3, height // 6, width // 3, height * 2 // 3
    else:
        x, y, w, h = cv2.boundingRect(points)
    margin = max(8, w // 6)
    left, right = max(0, x - margin), min(width, x + w + margin)
    top, bottom = max(0, y - margin), min(height, y + h + margin)
    if left >= right or top >= bottom:
        return
    layer = output[top:bottom, left:right].copy()
    span = bottom - top
    rng = np.random.default_rng(4242)
    for _ in range(34):
        line_x = int(rng.uniform(0, right - left))
        length = int(span * rng.uniform(0.25, 0.8))
        # Ogni linea scorre verso l'alto e ricompare dal basso.
        phase = (rng.uniform(0, 1) + progress * rng.uniform(2.0, 3.5)) % 1.0
        line_bottom = int((1.0 - phase) * (span + length))
        line_top = line_bottom - length
        thickness = int(rng.integers(1, 4))
        cv2.line(layer, (line_x, line_top), (line_x, line_bottom),
                 (20, 20, 20), thickness + 2, cv2.LINE_AA)
        cv2.line(layer, (line_x, line_top), (line_x, line_bottom),
                 (255, 250, 235), thickness, cv2.LINE_AA)
    region = output[top:bottom, left:right]
    region[:] = cv2.addWeighted(layer, strength, region, 1.0 - strength, 0)


def render_teleport(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                    progress: float, appearing: bool = False) -> np.ndarray:
    """Linee verticali sulla silhouette, lampo e sparizione (o il contrario)."""
    progress = min(1.0, max(0.0, progress))
    gone = 1.0 - progress if appearing else progress
    visibility = 1.0 - _smoothstep(0.35, 0.7, gone)
    flash = math.exp(-((gone - 0.45) / 0.14) ** 2) * 0.6
    alpha = (mask * visibility)[:, :, None]
    person = frame.astype(np.float32) * (1.0 - flash) + 255.0 * flash
    output = (person * alpha + background.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    strength = _smoothstep(0.0, 0.2, gone) * (1.0 - _smoothstep(0.75, 1.0, gone))
    if strength > 0.01:
        _speed_lines(output, mask, progress, strength)
    return output
