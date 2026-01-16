import math

import numpy as np

from .blocks import leaf_blocks, partition_frame


def intra_predict_frame(frame, mode="dc", angle=45.0, max_size=64, min_size=4, variance_threshold=100.0):
    luma = _to_luma(frame)
    luma_padded, orig_shape = _pad_to_multiple(luma, max_size)

    roots = partition_frame(
        frame,
        max_size=max_size,
        min_size=min_size,
        variance_threshold=variance_threshold,
    )
    leaves = leaf_blocks(roots)

    prediction = np.zeros_like(luma_padded)
    for block in leaves:
        pred = predict_block(luma_padded, block.x, block.y, block.size, mode, angle)
        prediction[block.y : block.y + block.size, block.x : block.x + block.size] = pred

    h, w = orig_shape
    return prediction[:h, :w]


def predict_block(luma, x, y, size, mode="dc", angle=45.0):
    top = _get_top_reference(luma, x, y, size)
    left = _get_left_reference(luma, x, y, size)

    if mode == "dc":
        values = []
        if top is not None:
            values.append(top.mean())
        if left is not None:
            values.append(left.mean())
        dc = np.mean(values) if values else 128.0
        return np.full((size, size), dc, dtype=np.float32)

    if mode == "horizontal":
        if left is None:
            return np.full((size, size), 128.0, dtype=np.float32)
        return np.repeat(left.reshape(size, 1), size, axis=1).astype(np.float32)

    if mode == "vertical":
        if top is None:
            return np.full((size, size), 128.0, dtype=np.float32)
        return np.repeat(top.reshape(1, size), size, axis=0).astype(np.float32)

    if mode == "angular":
        if top is None and left is None:
            return np.full((size, size), 128.0, dtype=np.float32)
        ref = top if top is not None else left
        slope = math.tan(math.radians(angle))
        pred = np.zeros((size, size), dtype=np.float32)
        for r in range(size):
            for c in range(size):
                idx = int(round(c + (r + 1) * slope))
                idx = max(0, min(size - 1, idx))
                pred[r, c] = ref[idx]
        return pred

    raise ValueError(f"Unknown intra mode: {mode}")


def _get_top_reference(luma, x, y, size):
    if y == 0:
        return None
    return luma[y - 1, x : x + size].astype(np.float32)


def _get_left_reference(luma, x, y, size):
    if x == 0:
        return None
    return luma[y : y + size, x - 1].astype(np.float32)


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
