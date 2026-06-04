import subprocess
import os

input_video = r""
output_dir = r""

os.makedirs(output_dir, exist_ok=True)

# 3 minutes = 180 seconds, 10 seconds per segment, 18 segments total
for i in range(18):
    start = i * 10
    end = start + 10
    output = f"{output_dir}/segment_{start}s_{end}s.mp4"

    cmd = [
        r"",
        "-y",
        "-ss", str(start),
        "-t", "10",
        "-i", input_video,
        "-c", "copy",
        output
    ]

    subprocess.run(cmd, check=True)
