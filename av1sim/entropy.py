import heapq
import struct
from collections import Counter


EOB_TOKEN = 0
ZRUN_TOKEN = 1
VAL_TOKEN = 2


def zigzag_indices(size):
    indices = []
    for s in range(2 * size - 1):
        if s % 2 == 0:
            r = min(s, size - 1)
            c = s - r
            while r >= 0 and c < size:
                indices.append((r, c))
                r -= 1
                c += 1
        else:
            c = min(s, size - 1)
            r = s - c
            while c >= 0 and r < size:
                indices.append((r, c))
                r += 1
                c -= 1
    return indices


def rle_encode_block(block, zz):
    flat = [block[r][c] for r, c in zz]
    tokens = []
    run = 0
    for val in flat:
        if val == 0:
            run += 1
            continue
        if run > 0:
            tokens.append((ZRUN_TOKEN, run))
            run = 0
        tokens.append((VAL_TOKEN, int(val)))
    tokens.append((EOB_TOKEN, 0))
    return tokens


def rle_decode_blocks(tokens, block_size, block_count):
    zz = zigzag_indices(block_size)
    blocks = []
    idx = 0
    for _ in range(block_count):
        out = [0] * (block_size * block_size)
        pos = 0
        while True:
            ttype, value = tokens[idx]
            idx += 1
            if ttype == EOB_TOKEN:
                break
            if ttype == ZRUN_TOKEN:
                pos += value
                continue
            if ttype == VAL_TOKEN:
                if pos < len(out):
                    out[pos] = value
                pos += 1
                continue
            raise ValueError("Unknown token type")
        block = [[0] * block_size for _ in range(block_size)]
        for i, (r, c) in enumerate(zz):
            block[r][c] = out[i]
        blocks.append(block)
    return blocks


def rle_encode_motion(vectors):
    tokens = []
    run = 0
    for dx, dy in vectors:
        if dx == 0 and dy == 0:
            run += 1
            continue
        if run > 0:
            tokens.append((ZRUN_TOKEN, run))
            run = 0
        packed = ((dx + 128) << 8) | (dy + 128)
        tokens.append((VAL_TOKEN, packed))
    tokens.append((EOB_TOKEN, 0))
    return tokens


def rle_decode_motion(tokens, count):
    vectors = []
    idx = 0
    while idx < len(tokens) and len(vectors) < count:
        ttype, value = tokens[idx]
        idx += 1
        if ttype == EOB_TOKEN:
            break
        if ttype == ZRUN_TOKEN:
            vectors.extend([(0, 0)] * value)
            continue
        if ttype == VAL_TOKEN:
            dx = (value >> 8) - 128
            dy = (value & 0xFF) - 128
            vectors.append((dx, dy))
            continue
        raise ValueError("Unknown token type")
    if len(vectors) < count:
        vectors.extend([(0, 0)] * (count - len(vectors)))
    return vectors


def build_huffman(symbols):
    freq = Counter(symbols)
    heap = []
    seq = 0
    for sym, count in freq.items():
        heap.append((count, seq, sym, None, None))
        seq += 1
    heapq.heapify(heap)

    if not heap:
        return {}, {}
    if len(heap) == 1:
        sym = heap[0][2]
        return {sym: (0, 1)}, {sym: 1}

    while len(heap) > 1:
        c1, _id1, s1, l1, r1 = heapq.heappop(heap)
        c2, _id2, s2, l2, r2 = heapq.heappop(heap)
        heapq.heappush(
            heap,
            (c1 + c2, seq, None, (c1, s1, l1, r1), (c2, s2, l2, r2)),
        )
        seq += 1

    root = (heap[0][0], heap[0][2], heap[0][3], heap[0][4])
    lengths = {}
    _assign_lengths(root, 0, lengths)
    codes = _canonical_codes(lengths)
    return codes, lengths


def _assign_lengths(node, depth, lengths):
    _count, sym, left, right = node
    if sym is not None:
        lengths[sym] = max(1, depth)
        return
    _assign_lengths(left, depth + 1, lengths)
    _assign_lengths(right, depth + 1, lengths)


def _canonical_codes(lengths):
    items = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    codes = {}
    code = 0
    prev_len = 0
    for sym, length in items:
        code <<= length - prev_len
        codes[sym] = (code, length)
        code += 1
        prev_len = length
    return codes


def encode_symbols(symbols, codes):
    bits = 0
    bit_len = 0
    out = bytearray()
    for sym in symbols:
        code, length = codes[sym]
        bits = (bits << length) | code
        bit_len += length
        while bit_len >= 8:
            shift = bit_len - 8
            out.append((bits >> shift) & 0xFF)
            bit_len -= 8
            bits &= (1 << shift) - 1
    if bit_len > 0:
        out.append(bits << (8 - bit_len))
    return bytes(out), bit_len


def decode_symbols(data, bit_len, lengths, symbol_count):
    table = _build_decode_table(lengths)
    result = []
    bit_pos = 0
    data_bits = len(data) * 8
    max_len = max(lengths.values()) if lengths else 1
    while bit_pos < data_bits and len(result) < symbol_count:
        code = 0
        for length in range(1, max_len + 1):
            if bit_pos >= data_bits:
                break
            byte_idx = bit_pos // 8
            bit_idx = 7 - (bit_pos % 8)
            bit = (data[byte_idx] >> bit_idx) & 1
            code = (code << 1) | bit
            bit_pos += 1
            key = (length, code)
            if key in table:
                result.append(table[key])
                break
    return result


def _build_decode_table(lengths):
    codes = _canonical_codes(lengths)
    return {(length, code): sym for sym, (code, length) in codes.items()}


def write_entropy_stream(handle, tokens):
    symbols = []
    symbol_map = {}
    ids = []
    for token in tokens:
        if token not in symbol_map:
            symbol_map[token] = len(symbols)
            symbols.append(token)
        ids.append(symbol_map[token])

    codes, lengths = build_huffman(ids)
    encoded, tail_bits = encode_symbols(ids, codes)

    handle.write(struct.pack("<H", len(symbols)))
    for idx, (ttype, value) in enumerate(symbols):
        length = lengths.get(idx, 1)
        handle.write(struct.pack("<BiB", ttype, int(value), length))
    handle.write(struct.pack("<I", len(ids)))
    handle.write(struct.pack("<I", len(encoded)))
    handle.write(struct.pack("<B", tail_bits))
    handle.write(encoded)


def read_entropy_stream(handle):
    symbol_count = struct.unpack("<H", handle.read(2))[0]
    symbols = []
    lengths = {}
    for idx in range(symbol_count):
        ttype, value, length = struct.unpack("<BiB", handle.read(6))
        symbols.append((ttype, value))
        lengths[idx] = length
    total_symbols = struct.unpack("<I", handle.read(4))[0]
    encoded_len = struct.unpack("<I", handle.read(4))[0]
    tail_bits = struct.unpack("<B", handle.read(1))[0]
    data = handle.read(encoded_len)
    decoded_ids = decode_symbols(data, tail_bits, lengths, total_symbols)
    return [symbols[idx] for idx in decoded_ids]
