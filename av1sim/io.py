import os

import cv2
import numpy as np

from .color import yuv420_to_bgr


def read_video_frames(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"No frames read from: {path}")

    return frames, fps


def write_video_frames(path, frames, fps):
    if not frames:
        raise ValueError("No frames to write")

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Unable to open video writer: {path}")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            raise ValueError("Frame size mismatch in output sequence")
        writer.write(frame)
    writer.release()


def read_yuv420_frames(path, width, height):
    if width <= 0 or height <= 0:
        raise ValueError("Width/height must be positive for YUV input")

    frame_size = width * height * 3 // 2
    with open(path, "rb") as handle:
        data = handle.read()

    if len(data) < frame_size:
        raise ValueError("YUV file too small for one frame")

    frame_count = len(data) // frame_size
    frames = []
    offset = 0
    for _ in range(frame_count):
        y = np.frombuffer(data, dtype=np.uint8, count=width * height, offset=offset)
        offset += width * height
        u = np.frombuffer(
            data,
            dtype=np.uint8,
            count=(width // 2) * (height // 2),
            offset=offset,
        )
        offset += (width // 2) * (height // 2)
        v = np.frombuffer(
            data,
            dtype=np.uint8,
            count=(width // 2) * (height // 2),
            offset=offset,
        )
        offset += (width // 2) * (height // 2)

        y = y.reshape(height, width)
        u = u.reshape(height // 2, width // 2)
        v = v.reshape(height // 2, width // 2)
        frames.append(yuv420_to_bgr(y, u, v))

    if len(data) % frame_size != 0:
        tail = len(data) - frame_count * frame_size
        raise ValueError(f"YUV file has {tail} trailing bytes (size mismatch)")

    return frames
