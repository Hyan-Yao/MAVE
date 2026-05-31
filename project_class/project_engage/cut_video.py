import subprocess
import os
import shutil

PROJECT_DIR = r"/Users/alyssa/Desktop/llm_as_a_judge/data/llm/project/Zoom Class Meeting Downing Soc 220 2⧸4⧸2021"
VIDEO_FILENAME = "Zoom Class Meeting Downing Soc 220 2⧸4⧸2021.mp4"
input_video = os.path.join(PROJECT_DIR, VIDEO_FILENAME)

output_root = os.path.join(PROJECT_DIR, "frames")
os.makedirs(output_root, exist_ok=True)

# macOS: use explicit Homebrew ffmpeg path
FFMPEG = "/opt/homebrew/bin/ffmpeg"

TOTAL_MINUTES = 80
INTERVAL_SEC = 10

if not os.path.isfile(input_video):
    raise FileNotFoundError(f"Video not found: {input_video}")
if shutil.which(FFMPEG) is None:
    raise FileNotFoundError(
        "ffmpeg not found in PATH. Install it with: brew install ffmpeg"
    )

for minute in range(TOTAL_MINUTES):
    minute_start = minute * 60
    minute_dir = os.path.join(output_root, f"minute_{minute:02d}")
    os.makedirs(minute_dir, exist_ok=True)

    for offset in range(0, 60, INTERVAL_SEC):
        t = minute_start + offset
        output_path = os.path.join(minute_dir, f"t_{t}s.jpg")

        cmd = [
            FFMPEG,
            "-y",
            "-ss", str(t),
            "-i", input_video,
            "-vframes", "1",
            "-q:v", "2",
            "-pix_fmt", "yuvj420p",
            output_path,
        ]

        subprocess.run(cmd, check=True)
