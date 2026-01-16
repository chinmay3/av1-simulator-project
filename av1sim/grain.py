import json

import cv2
import numpy as np


def analyze_grain(frames, bins=16, blur_ksize=5):
    sum_sq = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    for frame in frames:
        luma = _to_luma(frame)
        base = _denoise(luma, blur_ksize)
        residual = luma - base
        _accumulate_bins(luma, residual, bins, sum_sq, counts)

    strengths = _finalize_strengths(sum_sq, counts)
    edges = np.linspace(0, 256, bins + 1).tolist()
    return {"bins": bins, "edges": edges, "strengths": strengths.tolist()}


def synthesize_grain(luma, params, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    bins = params["bins"]
    edges = np.array(params["edges"], dtype=np.float32)
    strengths = np.array(params["strengths"], dtype=np.float32)

    idx = np.clip(np.digitize(luma, edges) - 1, 0, bins - 1)
    sigma = strengths[idx]
    noise = rng.normal(0.0, 1.0, size=luma.shape).astype(np.float32) * sigma
    return noise


def split_base_grain(frame, blur_ksize=5):
    luma = _to_luma(frame)
    base = _denoise(luma, blur_ksize)
    grain = luma - base
    return base, grain


def apply_grain_to_base(base_luma, params, rng=None):
    grain = synthesize_grain(base_luma, params, rng=rng)
    return base_luma + grain


def save_params(path, params):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2)


def load_params(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _accumulate_bins(luma, residual, bins, sum_sq, counts):
    edges = np.linspace(0, 256, bins + 1)
    idx = np.clip(np.digitize(luma, edges) - 1, 0, bins - 1)
    for b in range(bins):
        mask = idx == b
        if not np.any(mask):
            continue
        vals = residual[mask]
        sum_sq[b] += float(np.sum(vals * vals))
        counts[b] += int(vals.size)


def _finalize_strengths(sum_sq, counts):
    strengths = np.zeros_like(sum_sq, dtype=np.float64)
    valid = counts > 0
    strengths[valid] = np.sqrt(sum_sq[valid] / counts[valid])
    if not np.any(valid):
        return strengths
    mean_strength = float(np.mean(strengths[valid]))
    strengths[~valid] = mean_strength
    return strengths


def _denoise(luma, blur_ksize):
    k = max(1, int(blur_ksize))
    if k % 2 == 0:
        k += 1
    return cv2.GaussianBlur(luma, (k, k), 0)


def _to_luma(frame):
    if frame.ndim == 2:
        return frame.astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b
