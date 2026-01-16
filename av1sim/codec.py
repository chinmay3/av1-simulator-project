import json
import struct

import numpy as np

from .color import bgr_to_yuv420, yuv420_to_bgr
from .entropy import (
    EOB_TOKEN,
    VAL_TOKEN,
    ZRUN_TOKEN,
    rle_decode_blocks,
    rle_decode_motion,
    rle_encode_block,
    rle_encode_motion,
    write_entropy_stream,
    read_entropy_stream,
    zigzag_indices,
)
from .grain import analyze_grain, apply_grain_to_base


MAGIC = b"AV1S"
VERSION = 1


def encode_video(
    frames,
    fps,
    output_path,
    block_size=8,
    qp=20,
    min_qp=8,
    max_qp=40,
    target_kbps=0.0,
    search_range=8,
    mode="full",
    transform="dct",
    use_grain=False,
):
    if block_size % 2 != 0:
        raise ValueError("Block size must be even for 4:2:0")

    grain_params = analyze_grain(frames[: min(30, len(frames))]) if use_grain else None

    with open(output_path, "wb") as handle:
        _write_header(
            handle,
            frames[0].shape[1],
            frames[0].shape[0],
            fps,
            len(frames),
            block_size,
            transform,
            grain_params,
        )

        prev_rec = None
        current_qp = float(qp)
        for idx, frame in enumerate(frames):
            frame_type = 0 if idx == 0 else 1
            encoded = _encode_frame(
                frame,
                prev_rec,
                frame_type,
                block_size,
                current_qp,
                search_range,
                mode,
                transform,
            )
            handle.write(encoded["frame_header"])
            handle.write(encoded["mode_bytes"])
            handle.write(encoded["mv_stream"])
            for plane_stream in encoded["plane_streams"]:
                handle.write(plane_stream)

            prev_rec = encoded["recon_bgr"]
            if target_kbps > 0 and fps > 0:
                target_bits = target_kbps * 1000.0 / fps
                bits = encoded["estimated_bits"]
                if bits > target_bits:
                    current_qp = min(max_qp, current_qp + 1)
                else:
                    current_qp = max(min_qp, current_qp - 1)


def decode_video(input_path, output_path):
    frames, fps = decode_bitstream(input_path)
    from .io import write_video_frames

    write_video_frames(output_path, frames, fps)


def decode_bitstream(input_path):
    with open(input_path, "rb") as handle:
        header = _read_header(handle)
        frames = []
        prev_rec = None
        for _ in range(header["frame_count"]):
            frame_type, qp, mode_bytes = _read_frame_header(handle)
            modes = _unpack_modes(mode_bytes, header["blocks"])
            mv_tokens = read_entropy_stream(handle) if frame_type == 1 else []
            mv_list = rle_decode_motion(mv_tokens, header["blocks"])

            prev_planes = None
            if prev_rec is not None:
                prev_planes = bgr_to_yuv420(prev_rec)

            planes = []
            for plane in ("y", "u", "v"):
                tokens = read_entropy_stream(handle)
                plane_blocks = rle_decode_blocks(
                    tokens,
                    header["plane_sizes"][plane]["block"],
                    header["plane_sizes"][plane]["count"],
                )
                plane_recon = _reconstruct_plane(
                    plane_blocks,
                    header["plane_sizes"][plane],
                    prev_planes,
                    plane,
                    frame_type,
                    modes,
                    mv_list,
                    qp,
                    header["block_size"],
                    header["transform"],
                )
                planes.append(plane_recon)

            y, u, v = planes
            if header["grain_params"]:
                y = _apply_grain(y, header["grain_params"])
            bgr = yuv420_to_bgr(y, u, v)
            frames.append(bgr)
            prev_rec = bgr

    return frames, header["fps"]


def _encode_frame(
    frame,
    prev_rec,
    frame_type,
    block_size,
    qp,
    search_range,
    mode,
    transform,
):
    y, u, v = bgr_to_yuv420(frame)
    y_pad, y_shape = _pad_plane(y, block_size)
    uv_block = max(4, block_size // 2)
    u_pad, u_shape = _pad_plane(u, uv_block)
    v_pad, v_shape = _pad_plane(v, uv_block)

    blocks_w = y_pad.shape[1] // block_size
    blocks_h = y_pad.shape[0] // block_size
    block_count = blocks_w * blocks_h

    if frame_type == 1:
        prev_y, prev_u, prev_v = bgr_to_yuv420(prev_rec)
        prev_y, _ = _pad_plane(prev_y, block_size)
        prev_u, _ = _pad_plane(prev_u, uv_block)
        prev_v, _ = _pad_plane(prev_v, uv_block)
    else:
        prev_y = prev_u = prev_v = None

    modes = []
    mv_list = []
    rec_y = np.zeros_like(y_pad, dtype=np.float32)
    rec_u = np.zeros_like(u_pad, dtype=np.float32)
    rec_v = np.zeros_like(v_pad, dtype=np.float32)

    y_tokens = []
    u_tokens = []
    v_tokens = []
    zz_y = zigzag_indices(block_size)
    zz_uv = zigzag_indices(uv_block)

    for by in range(blocks_h):
        for bx in range(blocks_w):
            x = bx * block_size
            y0 = by * block_size
            curr_block = y_pad[y0 : y0 + block_size, x : x + block_size]

            if frame_type == 1:
                mv = _search_motion(curr_block, prev_y, x, y0, block_size, search_range, mode)
                inter_pred = _get_block(prev_y, x + mv[0], y0 + mv[1], block_size)
                intra_pred = _dc_predict(rec_y, x, y0, block_size)
                inter_cost = _sad(curr_block, inter_pred)
                intra_cost = _sad(curr_block, intra_pred)
                use_inter = inter_cost <= intra_cost
                modes.append(1 if use_inter else 0)
                mv_list.append(mv if use_inter else (0, 0))
                pred = inter_pred if use_inter else intra_pred
            else:
                modes.append(0)
                mv_list.append((0, 0))
                pred = _dc_predict(rec_y, x, y0, block_size)

            resid = curr_block.astype(np.float32) - pred.astype(np.float32)
            coeffs = _forward_transform(resid, transform)
            qcoeffs = _quantize(coeffs, qp)
            y_tokens.extend(rle_encode_block(qcoeffs, zz_y))
            recon_resid = _inverse_transform(_dequantize(qcoeffs, qp), transform)
            rec_y[y0 : y0 + block_size, x : x + block_size] = pred + recon_resid

            ux = bx * uv_block
            uy = by * uv_block
            if frame_type == 1 and modes[-1] == 1:
                mv_uv = (mv_list[-1][0] // 2, mv_list[-1][1] // 2)
                u_pred = _get_block(prev_u, ux + mv_uv[0], uy + mv_uv[1], uv_block)
                v_pred = _get_block(prev_v, ux + mv_uv[0], uy + mv_uv[1], uv_block)
            else:
                u_pred = _dc_predict(rec_u, ux, uy, uv_block)
                v_pred = _dc_predict(rec_v, ux, uy, uv_block)

            u_curr = u_pad[uy : uy + uv_block, ux : ux + uv_block]
            v_curr = v_pad[uy : uy + uv_block, ux : ux + uv_block]

            u_resid = u_curr.astype(np.float32) - u_pred.astype(np.float32)
            v_resid = v_curr.astype(np.float32) - v_pred.astype(np.float32)
            u_q = _quantize(_forward_transform(u_resid, transform), qp)
            v_q = _quantize(_forward_transform(v_resid, transform), qp)
            u_tokens.extend(rle_encode_block(u_q, zz_uv))
            v_tokens.extend(rle_encode_block(v_q, zz_uv))

            rec_u[uy : uy + uv_block, ux : ux + uv_block] = u_pred + _inverse_transform(
                _dequantize(u_q, qp), transform
            )
            rec_v[uy : uy + uv_block, ux : ux + uv_block] = v_pred + _inverse_transform(
                _dequantize(v_q, qp), transform
            )

    rec_y = np.clip(rec_y, 0, 255).astype(np.uint8)[: y_shape[0], : y_shape[1]]
    rec_u = np.clip(rec_u, 0, 255).astype(np.uint8)[: u_shape[0], : u_shape[1]]
    rec_v = np.clip(rec_v, 0, 255).astype(np.uint8)[: v_shape[0], : v_shape[1]]
    recon_bgr = yuv420_to_bgr(rec_y, rec_u, rec_v)

    mv_tokens = rle_encode_motion(_clamp_vectors(mv_list))
    mode_bytes = _pack_modes(modes)

    frame_header = struct.pack("<BBI", frame_type, int(round(qp)), len(mode_bytes))
    mv_stream = b""
    if frame_type == 1:
        mv_stream = _encode_stream(mv_tokens)

    y_stream = _encode_stream(y_tokens)
    u_stream = _encode_stream(u_tokens)
    v_stream = _encode_stream(v_tokens)

    estimated_bits = (
        (len(frame_header) + len(mode_bytes) + len(mv_stream) + len(y_stream) + len(u_stream) + len(v_stream))
        * 8
    )

    return {
        "frame_header": frame_header,
        "mode_bytes": mode_bytes,
        "mv_stream": mv_stream,
        "plane_streams": [y_stream, u_stream, v_stream],
        "recon_bgr": recon_bgr,
        "estimated_bits": estimated_bits,
    }


def _reconstruct_plane(
    blocks,
    plane_info,
    prev_planes,
    plane,
    frame_type,
    modes,
    mv_list,
    qp,
    block_size,
    transform,
):
    width = plane_info["width"]
    height = plane_info["height"]
    blk = plane_info["block"]
    blocks_w = width // blk
    blocks_h = height // blk

    rec = np.zeros((height, width), dtype=np.float32)
    idx = 0
    for by in range(blocks_h):
        for bx in range(blocks_w):
            block = np.array(blocks[idx], dtype=np.float32)
            idx += 1
            x = bx * blk
            y0 = by * blk

            if frame_type == 1:
                mode_bit = modes[by * blocks_w + bx]
                mv = mv_list[by * blocks_w + bx]
                if plane != "y":
                    mv = (mv[0] // 2, mv[1] // 2)
                pred = _inter_or_intra(rec, prev_planes, mv, x, y0, blk, mode_bit, plane)
            else:
                pred = _dc_predict(rec, x, y0, blk)

            recon = pred + _inverse_transform(_dequantize(block, qp), transform)
            rec[y0 : y0 + blk, x : x + blk] = recon

    return np.clip(rec, 0, 255).astype(np.uint8)


def _inter_or_intra(rec_plane, prev_planes, mv, x, y0, size, mode_bit, plane):
    if mode_bit == 1 and prev_planes is not None:
        prev_plane = {"y": prev_planes[0], "u": prev_planes[1], "v": prev_planes[2]}[plane]
        return _get_block(prev_plane, x + mv[0], y0 + mv[1], size)
    return _dc_predict(rec_plane, x, y0, size)


def _search_motion(curr_block, ref_plane, x, y0, size, search_range, mode):
    h, w = ref_plane.shape
    best_dx = 0
    best_dy = 0
    best_cost = float("inf")

    if mode == "diamond":
        best_cost = _sad(curr_block, _get_block(ref_plane, x, y0, size))
        while True:
            improved = False
            for dx, dy in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                cand_dx = best_dx + dx
                cand_dy = best_dy + dy
                if abs(cand_dx) > search_range or abs(cand_dy) > search_range:
                    continue
                cx = x + cand_dx
                cy = y0 + cand_dy
                if cx < 0 or cy < 0 or cx + size > w or cy + size > h:
                    continue
                cand = _get_block(ref_plane, cx, cy, size)
                cost = _sad(curr_block, cand)
                if cost < best_cost:
                    best_cost = cost
                    best_dx = cand_dx
                    best_dy = cand_dy
                    improved = True
            if not improved:
                break
        return best_dx, best_dy

    for dy in range(-search_range, search_range + 1):
        cy = y0 + dy
        if cy < 0 or cy + size > h:
            continue
        for dx in range(-search_range, search_range + 1):
            cx = x + dx
            if cx < 0 or cx + size > w:
                continue
            cand = _get_block(ref_plane, cx, cy, size)
            cost = _sad(curr_block, cand)
            if cost < best_cost:
                best_cost = cost
                best_dx = dx
                best_dy = dy
    return best_dx, best_dy


def _write_header(handle, width, height, fps, frame_count, block_size, transform, grain_params):
    flags = 1 if grain_params else 0
    transform_id = 0 if transform == "dct" else 1
    header = struct.pack(
        "<4sBHHfIBBB",
        MAGIC,
        VERSION,
        width,
        height,
        float(fps),
        frame_count,
        block_size,
        flags,
        transform_id,
    )
    handle.write(header)
    if grain_params:
        payload = json.dumps(grain_params).encode("utf-8")
        handle.write(struct.pack("<I", len(payload)))
        handle.write(payload)


def _read_header(handle):
    magic, version, width, height, fps, frame_count, block_size, flags, transform_id = struct.unpack(
        "<4sBHHfIBBB", handle.read(20)
    )
    if magic != MAGIC:
        raise ValueError("Not an AV1SIM bitstream")
    grain_params = None
    if flags & 1:
        length = struct.unpack("<I", handle.read(4))[0]
        grain_params = json.loads(handle.read(length).decode("utf-8"))
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "block_size": block_size,
        "grain_params": grain_params,
        "blocks": (width + block_size - 1) // block_size * ((height + block_size - 1) // block_size),
        "plane_sizes": _plane_sizes(width, height, block_size),
        "transform": "dct" if transform_id == 0 else "wht",
    }


def _read_frame_header(handle):
    frame_type, qp, mode_len = struct.unpack("<BBI", handle.read(6))
    mode_bytes = handle.read(mode_len)
    return frame_type, qp, mode_bytes


def _encode_stream(tokens):
    from io import BytesIO

    buffer = BytesIO()
    write_entropy_stream(buffer, tokens)
    return buffer.getvalue()


def _pack_modes(modes):
    out = bytearray((len(modes) + 7) // 8)
    for i, mode in enumerate(modes):
        if mode:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def _unpack_modes(data, count):
    modes = []
    for i in range(count):
        byte = data[i // 8] if data else 0
        modes.append((byte >> (i % 8)) & 1)
    return modes


def _apply_grain(luma, params):
    grain = apply_grain_to_base(luma.astype(np.float32), params, rng=np.random.default_rng(0))
    return np.clip(grain, 0, 255).astype(np.uint8)


def _forward_transform(block, mode):
    import cv2

    block_f = block.astype(np.float32)
    if mode == "wht":
        return _hadamard2(block_f)
    return cv2.dct(block_f)


def _inverse_transform(coeffs, mode):
    import cv2

    if mode == "wht":
        return _hadamard2(coeffs) / coeffs.shape[0] / coeffs.shape[1]
    return cv2.idct(coeffs)


def _quantize(coeffs, qp):
    qstep = max(1.0, float(qp))
    return np.round(coeffs / qstep).astype(np.int32)


def _dequantize(qcoeffs, qp):
    qstep = max(1.0, float(qp))
    return qcoeffs.astype(np.float32) * qstep


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


def _pad_plane(plane, block):
    height, width = plane.shape
    pad_h = (block - (height % block)) % block
    pad_w = (block - (width % block)) % block
    if pad_h == 0 and pad_w == 0:
        return plane.astype(np.float32), (height, width)
    padded = np.pad(plane, ((0, pad_h), (0, pad_w)), mode="edge").astype(np.float32)
    return padded, (height, width)


def _plane_sizes(width, height, block):
    uv_block = max(4, block // 2)
    y_w = ((width + block - 1) // block) * block
    y_h = ((height + block - 1) // block) * block
    u_w = ((width // 2 + uv_block - 1) // uv_block) * uv_block
    u_h = ((height // 2 + uv_block - 1) // uv_block) * uv_block
    return {
        "y": {"width": y_w, "height": y_h, "block": block, "count": (y_w // block) * (y_h // block)},
        "u": {
            "width": u_w,
            "height": u_h,
            "block": uv_block,
            "count": (u_w // uv_block) * (u_h // uv_block),
        },
        "v": {
            "width": u_w,
            "height": u_h,
            "block": uv_block,
            "count": (u_w // uv_block) * (u_h // uv_block),
        },
    }


def _get_block(plane, x, y0, size):
    return plane[y0 : y0 + size, x : x + size].astype(np.float32)


def _dc_predict(plane, x, y0, size):
    top = plane[y0 - 1, x : x + size] if y0 > 0 else None
    left = plane[y0 : y0 + size, x - 1] if x > 0 else None
    values = []
    if top is not None:
        values.append(float(np.mean(top)))
    if left is not None:
        values.append(float(np.mean(left)))
    dc = float(np.mean(values)) if values else 128.0
    return np.full((size, size), dc, dtype=np.float32)


def _sad(a, b):
    return float(np.sum(np.abs(a - b)))


def _clamp_vectors(mv_list):
    clamped = []
    for dx, dy in mv_list:
        dx = max(-127, min(127, int(dx)))
        dy = max(-127, min(127, int(dy)))
        clamped.append((dx, dy))
    return clamped
