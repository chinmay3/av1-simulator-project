import numpy as np

from .blocks import leaf_blocks, partition_frame


def transform_quant_frame(
    frame,
    mode="dct",
    qp=20,
    qp_mode="constant",
    min_qp=10,
    max_qp=40,
    variance_threshold=100.0,
    max_size=64,
    min_size=4,
    var_min=50.0,
    var_max=500.0,
):
    luma = _to_luma(frame)
    luma_pad, orig_shape = _pad_to_multiple(luma, max_size)

    roots = partition_frame(
        frame,
        max_size=max_size,
        min_size=min_size,
        variance_threshold=variance_threshold,
    )
    leaves = leaf_blocks(roots)

    recon = np.zeros_like(luma_pad)
    for block in leaves:
        block_luma = luma_pad[block.y : block.y + block.size, block.x : block.x + block.size]
        block_qp = _select_qp(
            block.variance,
            qp_mode,
            qp,
            min_qp,
            max_qp,
            var_min,
            var_max,
        )
        coeffs = _forward_transform(block_luma, mode)
        qcoeffs = _quantize(coeffs, block_qp)
        deq = _dequantize(qcoeffs, block_qp)
        recon_block = _inverse_transform(deq, mode)
        recon[block.y : block.y + block.size, block.x : block.x + block.size] = recon_block

    h, w = orig_shape
    return recon[:h, :w]


def _forward_transform(block, mode):
    block_f = block.astype(np.float32) - 128.0
    if mode == "dct":
        return _dct2(block_f)
    if mode == "wht":
        return _hadamard2(block_f)
    raise ValueError(f"Unknown transform mode: {mode}")


def _inverse_transform(coeffs, mode):
    if mode == "dct":
        recon = _idct2(coeffs)
    elif mode == "wht":
        recon = _hadamard2(coeffs) / coeffs.shape[0] / coeffs.shape[1]
    else:
        raise ValueError(f"Unknown transform mode: {mode}")
    return recon + 128.0


def _quantize(coeffs, qp):
    qstep = max(1.0, float(qp))
    return np.round(coeffs / qstep)


def _dequantize(qcoeffs, qp):
    qstep = max(1.0, float(qp))
    return qcoeffs * qstep


def _select_qp(variance, qp_mode, qp, min_qp, max_qp, var_min, var_max):
    if qp_mode == "constant":
        return qp
    if qp_mode != "adaptive":
        raise ValueError(f"Unknown qp mode: {qp_mode}")
    if var_max <= var_min:
        return qp
    t = (variance - var_min) / (var_max - var_min)
    t = min(1.0, max(0.0, t))
    return max_qp - t * (max_qp - min_qp)


def _dct2(block):
    import cv2

    return cv2.dct(block.astype(np.float32))


def _idct2(coeffs):
    import cv2

    return cv2.idct(coeffs.astype(np.float32))


def _hadamard2(block):
    data = block.astype(np.float32)
    data = _hadamard_axis(data, axis=0)
    data = _hadamard_axis(data, axis=1)
    return data


def _hadamard_axis(data, axis):
    n = data.shape[axis]
    if n & (n - 1) != 0:
        raise ValueError("Hadamard transform requires power-of-two size")
    h = 1
    out = data.copy()
    while h < n:
        step = h * 2
        slices = [slice(None)] * out.ndim
        for i in range(0, n, step):
            slices[axis] = slice(i, i + h)
            a = out[tuple(slices)]
            slices[axis] = slice(i + h, i + step)
            b = out[tuple(slices)]
            out[tuple(slices)] = a - b
            slices[axis] = slice(i, i + h)
            out[tuple(slices)] = a + b
        h = step
    return out


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
