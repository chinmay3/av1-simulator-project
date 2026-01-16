import json
from dataclasses import dataclass

import numpy as np

from .blocks import leaf_blocks, partition_frame


@dataclass
class MotionVector:
    x: int
    y: int
    size: int
    dx: int
    dy: int
    cost: float


def inter_predict_frame(
    curr_frame,
    ref_frame,
    mode="full",
    search_range=8,
    max_size=64,
    min_size=4,
    variance_threshold=100.0,
):
    curr_luma = _to_luma(curr_frame)
    ref_luma = _to_luma(ref_frame)

    curr_pad, orig_shape = _pad_to_multiple(curr_luma, max_size)
    ref_pad, _ = _pad_to_multiple(ref_luma, max_size)

    roots = partition_frame(
        curr_frame,
        max_size=max_size,
        min_size=min_size,
        variance_threshold=variance_threshold,
    )
    leaves = leaf_blocks(roots)

    predicted = np.zeros_like(curr_pad)
    residual = np.zeros_like(curr_pad)
    vectors = []
    for block in leaves:
        mv = _match_block(
            curr_pad,
            ref_pad,
            block.x,
            block.y,
            block.size,
            mode,
            search_range,
        )
        vectors.append(mv)
        pred_block = ref_pad[
            mv.y + mv.dy : mv.y + mv.dy + mv.size,
            mv.x + mv.dx : mv.x + mv.dx + mv.size,
        ]
        predicted[block.y : block.y + block.size, block.x : block.x + block.size] = pred_block
        curr_block = curr_pad[block.y : block.y + block.size, block.x : block.x + block.size]
        residual[block.y : block.y + block.size, block.x : block.x + block.size] = (
            curr_block - pred_block
        )

    h, w = orig_shape
    return predicted[:h, :w], residual[:h, :w], vectors


def serialize_motion_vectors(vectors):
    return json.dumps([mv.__dict__ for mv in vectors])


def _match_block(curr_luma, ref_luma, x, y, size, mode, search_range):
    if mode == "diamond":
        return _diamond_search(curr_luma, ref_luma, x, y, size, search_range)
    if mode == "full":
        return _full_search(curr_luma, ref_luma, x, y, size, search_range)
    raise ValueError(f"Unknown inter mode: {mode}")


def _full_search(curr_luma, ref_luma, x, y, size, search_range):
    h, w = ref_luma.shape
    curr_block = curr_luma[y : y + size, x : x + size]
    best_cost = float("inf")
    best_dx = 0
    best_dy = 0
    for dy in range(-search_range, search_range + 1):
        ry = y + dy
        if ry < 0 or ry + size > h:
            continue
        for dx in range(-search_range, search_range + 1):
            rx = x + dx
            if rx < 0 or rx + size > w:
                continue
            cand = ref_luma[ry : ry + size, rx : rx + size]
            cost = _sad(curr_block, cand)
            if cost < best_cost:
                best_cost = cost
                best_dx = dx
                best_dy = dy
    return MotionVector(x, y, size, best_dx, best_dy, best_cost)


def _diamond_search(curr_luma, ref_luma, x, y, size, search_range):
    h, w = ref_luma.shape
    curr_block = curr_luma[y : y + size, x : x + size]
    best_dx = 0
    best_dy = 0
    best_cost = _sad(curr_block, ref_luma[y : y + size, x : x + size])

    while True:
        improved = False
        for step_dx, step_dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
            cand_dx = best_dx + step_dx
            cand_dy = best_dy + step_dy
            if abs(cand_dx) > search_range or abs(cand_dy) > search_range:
                continue
            rx = x + cand_dx
            ry = y + cand_dy
            if rx < 0 or ry < 0 or rx + size > w or ry + size > h:
                continue
            cand = ref_luma[ry : ry + size, rx : rx + size]
            cost = _sad(curr_block, cand)
            if cost < best_cost:
                best_cost = cost
                best_dx = cand_dx
                best_dy = cand_dy
                improved = True
        if not improved:
            break
    return MotionVector(x, y, size, best_dx, best_dy, best_cost)


def _sad(a, b):
    return float(np.sum(np.abs(a - b)))


def _to_luma(frame):
    if frame.ndim == 2:
        return frame.astype(np.float32)
    b = frame[:, :, 0].astype(np.float32)
    g = frame[:, :, 1].astype(np.float32)
    r = frame[:, :, 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _pad_to_multiple(luma, block_size):
    height, width = luma.shape
    pad_h = (block_size - (height % block_size)) % block_size
    pad_w = (block_size - (width % block_size)) % block_size
    if pad_h == 0 and pad_w == 0:
        return luma, (height, width)
    padded = np.pad(luma, ((0, pad_h), (0, pad_w)), mode="edge")
    return padded, (height, width)
