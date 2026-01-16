import os
import threading
import time

import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from av1sim.codec import decode_video, encode_video
from av1sim.io import read_yuv420_frames


app = Flask(__name__, static_folder="static", template_folder="templates")

BASE_DIR = os.getcwd()
OUTPUT_DIR = "/tmp" if os.getenv("VERCEL") else BASE_DIR

CONFIG = {
    "input_yuv": os.path.join(BASE_DIR, "input.yuv"),
    "yuv_width": 352,
    "yuv_height": 288,
    "yuv_fps": 30.0,
    "yuv_frames": 60,
    "output_av1s": os.path.join(OUTPUT_DIR, "output.av1s"),
    "output_mp4": os.path.join(OUTPUT_DIR, "decoded.mp4"),
    "block": 8,
    "qp": 20,
    "search": 8,
    "mode": "diamond",
    "transform": "dct",
}

JOBS = {
    "compress": {"state": "idle", "started": 0.0, "error": "", "output": 0},
    "decode": {"state": "idle", "started": 0.0, "error": "", "output": 0},
}

LOCK = threading.Lock()


def _file_size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


def _human_size(num):
    if num <= 0:
        return "0 B"
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def _run_compress():
    with LOCK:
        JOBS["compress"].update({"state": "running", "started": time.time(), "error": "", "output": 0})
    try:
        frames = _load_frames()
        encode_video(
            frames,
            CONFIG["yuv_fps"],
            CONFIG["output_av1s"],
            block_size=CONFIG["block"],
            qp=CONFIG["qp"],
            search_range=CONFIG["search"],
            mode=CONFIG["mode"],
            transform=CONFIG["transform"],
        )
        size = _file_size(CONFIG["output_av1s"])
        with LOCK:
            JOBS["compress"].update({"state": "done", "output": size})
    except Exception as exc:
        with LOCK:
            JOBS["compress"].update({"state": "error", "error": str(exc)})


def _run_decode():
    with LOCK:
        JOBS["decode"].update({"state": "running", "started": time.time(), "error": "", "output": 0})
    try:
        decode_video(CONFIG["output_av1s"], CONFIG["output_mp4"])
        size = _file_size(CONFIG["output_mp4"])
        with LOCK:
            JOBS["decode"].update({"state": "done", "output": size})
    except Exception as exc:
        with LOCK:
            JOBS["decode"].update({"state": "error", "error": str(exc)})


def _start_job(job_name, target):
    with LOCK:
        if JOBS[job_name]["state"] == "running":
            return False
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    return True


@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/db")
def db():
    yuv_size = _source_size()
    yuv_name = os.path.basename(CONFIG["input_yuv"]) if os.path.exists(CONFIG["input_yuv"]) else "sample.yuv"
    return render_template(
        "db.html",
        yuv_name=yuv_name,
        yuv_size=_human_size(yuv_size),
    )


@app.route("/compress")
def compress_page():
    return render_template("compress.html")


@app.route("/transfer")
def transfer_page():
    av1s_size = _human_size(_file_size(CONFIG["output_av1s"]))
    return render_template("transfer.html", av1s_size=av1s_size)


@app.route("/decode")
def decode_page():
    return render_template("decode.html")


@app.route("/api/compress", methods=["POST"])
def api_compress():
    started = _start_job("compress", _run_compress)
    return jsonify({"started": started})


@app.route("/api/decode", methods=["POST"])
def api_decode():
    started = _start_job("decode", _run_decode)
    return jsonify({"started": started})


@app.route("/api/status/<job_name>")
def api_status(job_name):
    if job_name not in JOBS:
        return jsonify({"error": "unknown job"}), 404
    job = JOBS[job_name]
    state = job["state"]
    progress = 0
    if state == "running":
        elapsed = time.time() - job["started"]
        target = 18.0 if job_name == "compress" else 10.0
        progress = min(95, int((elapsed / target) * 100))
    if state == "done":
        progress = 100
    return jsonify(
        {
            "state": state,
            "progress": progress,
            "error": job["error"],
            "output": _human_size(job["output"]),
        }
    )


@app.route("/media/decoded.mp4")
def media_decoded():
    if not os.path.exists(CONFIG["output_mp4"]):
        return "", 404
    return send_file(CONFIG["output_mp4"], mimetype="video/mp4")


def _load_frames():
    if os.path.exists(CONFIG["input_yuv"]):
        return read_yuv420_frames(
            CONFIG["input_yuv"], CONFIG["yuv_width"], CONFIG["yuv_height"]
        )
    return _generate_frames(CONFIG["yuv_frames"], CONFIG["yuv_width"], CONFIG["yuv_height"])


def _source_size():
    if os.path.exists(CONFIG["input_yuv"]):
        return _file_size(CONFIG["input_yuv"])
    return int(CONFIG["yuv_frames"] * CONFIG["yuv_width"] * CONFIG["yuv_height"] * 3 // 2)


def _generate_frames(count, width, height):
    frames = []
    for i in range(count):
        x = np.linspace(0, 255, width, dtype=np.float32)
        y = np.linspace(0, 255, height, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        base = (xv * 0.6 + yv * 0.4 + (i * 3) % 255) % 255
        bgr = np.stack([base, np.roll(base, i % width, axis=1), np.flipud(base)], axis=2)
        frames.append(bgr.astype(np.uint8))
    return frames


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)
