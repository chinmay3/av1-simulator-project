from .color import bgr_to_yuv420, yuv420_to_bgr
from .io import read_video_frames, write_video_frames


def run_baseline_pipeline(input_path, output_path):
    frames, fps = read_video_frames(input_path)

    reconstructed = []
    for frame in frames:
        y, u, v = bgr_to_yuv420(frame)
        recon = yuv420_to_bgr(y, u, v)
        reconstructed.append(recon)

    write_video_frames(output_path, reconstructed, fps)
