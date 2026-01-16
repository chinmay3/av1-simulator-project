from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


@dataclass
class Block:
    x: int
    y: int
    size: int
    variance: float
    children: List["Block"]

    @property
    def is_leaf(self) -> bool:
        return not self.children


def partition_frame(frame, max_size=64, min_size=4, variance_threshold=100.0):
    luma = _to_luma(frame)
    luma_padded = _pad_to_multiple(luma, max_size)

    blocks = []
    height, width = luma_padded.shape
    for y in range(0, height, max_size):
        for x in range(0, width, max_size):
            blocks.append(
                _split_block(
                    luma_padded,
                    x,
                    y,
                    max_size,
                    min_size,
                    variance_threshold,
                )
            )
    return blocks


def leaf_blocks(blocks):
    leaves = []
    for block in blocks:
        leaves.extend(_collect_leaves(block))
    return leaves


def draw_block_overlay(frame, leaves, color=(0, 200, 255), thickness=1):
    overlay = frame.copy()
    for block in leaves:
        x0, y0 = block.x, block.y
        x1, y1 = block.x + block.size - 1, block.y + block.size - 1
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, thickness)
    return overlay


def _split_block(luma, x, y, size, min_size, variance_threshold):
    region = luma[y : y + size, x : x + size]
    variance = float(np.var(region))

    if size <= min_size or variance <= variance_threshold or size % 2 != 0:
        return Block(x, y, size, variance, [])

    half = size // 2
    if half < min_size:
        return Block(x, y, size, variance, [])

    children = [
        _split_block(luma, x, y, half, min_size, variance_threshold),
        _split_block(luma, x + half, y, half, min_size, variance_threshold),
        _split_block(luma, x, y + half, half, min_size, variance_threshold),
        _split_block(luma, x + half, y + half, half, min_size, variance_threshold),
    ]
    return Block(x, y, size, variance, children)


def _collect_leaves(block):
    if block.is_leaf:
        return [block]
    leaves = []
    for child in block.children:
        leaves.extend(_collect_leaves(child))
    return leaves


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
        return luma
    return np.pad(luma, ((0, pad_h), (0, pad_w)), mode="edge")
