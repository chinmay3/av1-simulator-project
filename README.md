# AV1 Simulator (Part 1)

Baseline pipeline: read video -> convert to YUV 4:2:0 -> reconstruct -> write video.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python -m scripts.run_pipeline input.mp4 output.mp4
```

## Notes
- Uses BT.601 style YUV conversion for learning purposes.
- Subsamples chroma with 4:2:0 (2x2 averaging) and upsamples with nearest.

## Part 2: Block partitioning
The block partitioner builds a quadtree over luma to simulate adaptive block sizes.

```python
from av1sim.blocks import partition_frame, leaf_blocks

blocks = partition_frame(frame, max_size=64, min_size=4, variance_threshold=100.0)
leaves = leaf_blocks(blocks)
```

Visualize the block grid:

```bash
python -m scripts.visualize_blocks input.mp4 blocks.mp4 --variance 120
```

## Part 3: Intra prediction
Predicts blocks from spatial neighbors and outputs a grayscale prediction video.

```bash
python -m scripts.intra_predict input.mp4 pred.mp4 --mode dc
python -m scripts.intra_predict input.mp4 pred.mp4 --mode horizontal
python -m scripts.intra_predict input.mp4 pred.mp4 --mode vertical
python -m scripts.intra_predict input.mp4 pred.mp4 --mode angular --angle 35
```

## Part 4: Inter prediction
Predicts each frame from the previous frame using block matching and outputs a grayscale prediction video and optional residuals.

```bash
python -m scripts.inter_predict input.mp4 pred.mp4 --mode full --search 8
python -m scripts.inter_predict input.mp4 pred.mp4 --mode diamond --search 8 --residual-output residual.mp4
python -m scripts.inter_predict input.mp4 pred.mp4 --mv-output motion.jsonl
python -m scripts.inter_predict input.mp4 pred.mp4 --combo-output combo.mp4
```

## Part 5: Transform + quantization
Applies block transforms and quantization to show the bitrate/quality tradeoff.

```bash
python -m scripts.transform_quant input.mp4 tq.mp4 --mode dct --qp 20
python -m scripts.transform_quant input.mp4 tq.mp4 --mode wht --qp 24 --qp-mode adaptive
python -m scripts.transform_quant input.mp4 tq.mp4 --mode dct --qp 12 --combo-output combo.mp4
```

## Part 6: Film grain analysis + synthesis
Analyzes grain strength vs luma and synthesizes grain on a denoised base.

```bash
python -m scripts.film_grain input.mp4 --base-output base.mp4 --synth-output grain.mp4 --params-output grain.json
python -m scripts.film_grain input.mp4 --combo-output combo.mp4
```

## Part 7-9: Entropy coding, rate control, decoder
Encodes frames to a compact bitstream with RLE + Huffman and decodes it back to video.

```bash
python -m scripts.encode input.mp4 output.av1s --block 8 --qp 20 --mode diamond --target-kbps 800
python -m scripts.decode output.av1s decoded.mp4
```

Raw YUV input (4:2:0):

```bash
python -m scripts.encode input.yuv output.av1s --yuv-width 176 --yuv-height 144 --yuv-fps 15
```

## Metrics + RD curve
Compute compression ratio and a PSNR-based RD curve.

```bash
python -m scripts.metrics input.yuv --yuv-width 352 --yuv-height 288 --yuv-fps 30 --qp-list 12,16,20,24,28 --output-csv rd_curve.csv
python -m scripts.metrics input.yuv --yuv-width 352 --yuv-height 288 --yuv-fps 30 --bitstream output.av1s
```

## Netflix-style demo UI (Flask)
Run a local web UI that drives the encoder/decoder.

```bash
python app.py
```

Place a `thumbnail.png` image in `static/thumbnail.png` and open `http://127.0.0.1:5000`.
