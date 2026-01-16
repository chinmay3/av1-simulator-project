import numpy as np


def bgr_to_yuv420(frame_bgr):
    b = frame_bgr[:, :, 0].astype(np.float32)
    g = frame_bgr[:, :, 1].astype(np.float32)
    r = frame_bgr[:, :, 2].astype(np.float32)

    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.169 * r - 0.331 * g + 0.5 * b + 128.0
    v = 0.5 * r - 0.419 * g - 0.081 * b + 128.0

    y = np.clip(y, 0, 255).astype(np.uint8)
    u = np.clip(u, 0, 255)
    v = np.clip(v, 0, 255)

    u_sub = _subsample_420(u)
    v_sub = _subsample_420(v)

    return y, u_sub, v_sub


def yuv420_to_bgr(y, u_sub, v_sub):
    u = _upsample_420(u_sub, y.shape)
    v = _upsample_420(v_sub, y.shape)

    y_f = y.astype(np.float32)
    u_f = u.astype(np.float32) - 128.0
    v_f = v.astype(np.float32) - 128.0

    r = y_f + 1.403 * v_f
    g = y_f - 0.344 * u_f - 0.714 * v_f
    b = y_f + 1.770 * u_f

    bgr = np.stack(
        [
            np.clip(b, 0, 255),
            np.clip(g, 0, 255),
            np.clip(r, 0, 255),
        ],
        axis=2,
    ).astype(np.uint8)

    return bgr


def _subsample_420(channel):
    h, w = channel.shape
    h2 = h // 2 * 2
    w2 = w // 2 * 2
    channel = channel[:h2, :w2]
    return (
        channel.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3)).astype(np.uint8)
    )


def _upsample_420(channel_sub, target_shape):
    h, w = target_shape
    return np.repeat(np.repeat(channel_sub, 2, axis=0), 2, axis=1)[:h, :w]
