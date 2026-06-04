import cv2
import numpy as np
import os
import sys
from pathlib import Path

print("USING PYTHON:", sys.executable)


def read_image_unicode_safe(path):
    """Read image bytes in Python then decode via OpenCV to avoid Unicode path issues."""
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def write_image_unicode_safe(path, img):
    """Encode with OpenCV and write bytes in Python to avoid cv2.imwrite Unicode path failures."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False


# macOS paths (change VIDEO_FOLDER for another class/video)
PROJECT_LLM_DIR = Path("")
VIDEO_FOLDER = ""
FRAMES_ROOT = str(PROJECT_LLM_DIR / "project" / VIDEO_FOLDER / "frames")
OUT_DIR = str(PROJECT_LLM_DIR / "project" / VIDEO_FOLDER / "images")

# Fixed grid split configuration (rollback version)
GRID_ROWS = 5
GRID_COLS = 5

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(FRAMES_ROOT):
    raise RuntimeError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")

# One folder per minute: minute_00, minute_01, ...
minute_dirs = sorted(
    [
        d
        for d in os.listdir(FRAMES_ROOT)
        if os.path.isdir(os.path.join(FRAMES_ROOT, d)) and d.startswith("minute_")
    ]
)

print(f"FRAMES_ROOT: {FRAMES_ROOT}")
print(
    f"Found {len(minute_dirs)} minute_* folders: "
    f"{minute_dirs[:5]}{'...' if len(minute_dirs) > 5 else ''}"
)

if not minute_dirs:
    raise RuntimeError(f"No minute_* folders found in {FRAMES_ROOT}")

total_written = 0
for minute_id in minute_dirs:
    minute_path = os.path.join(FRAMES_ROOT, minute_id)
    image_files = sorted(
        [
            f
            for f in os.listdir(minute_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    )

    if not image_files:
        print(f"[SKIP] {minute_id}: no images in {minute_path}")
        continue

    print(f"\nProcessing {minute_id}: {len(image_files)} images")

    minute_out_base = os.path.join(OUT_DIR, minute_id)
    os.makedirs(minute_out_base, exist_ok=True)

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            gid = r * GRID_COLS + c
            os.makedirs(os.path.join(minute_out_base, f"grid_{gid:02d}"), exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(minute_path, img_name)
        frame = read_image_unicode_safe(img_path)
        if frame is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue
        if frame.ndim != 3:
            continue

        h, w, _ = frame.shape
        if h == 0 or w == 0:
            continue

        time_tag = os.path.splitext(img_name)[0]
        cell_h = h // GRID_ROWS
        cell_w = w // GRID_COLS

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                y1 = r * cell_h
                y2 = h if r == GRID_ROWS - 1 else (r + 1) * cell_h
                x1 = c * cell_w
                x2 = w if c == GRID_COLS - 1 else (c + 1) * cell_w

                if y2 <= y1 or x2 <= x1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                gid = r * GRID_COLS + c
                grid_dir = os.path.join(minute_out_base, f"grid_{gid:02d}")
                out_path = os.path.join(grid_dir, f"{time_tag}.jpg")
                if write_image_unicode_safe(out_path, crop):
                    total_written += 1

print(f"\nFinished. Total images written: {total_written}")
print(f"Output directory: {OUT_DIR}")
import cv2
import numpy as np
import os
import sys
import traceback
from pathlib import Path

print("USING PYTHON:", sys.executable)


def read_image_unicode_safe(path):
    """Read image bytes in Python then decode via OpenCV to avoid Unicode path issues in OpenCV C++ layer."""
    try:
        with open(path, "rb") as f:
            buf = f.read()
        arr = np.frombuffer(buf, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception:
        return None


def write_image_unicode_safe(path, img):
    """Encode with OpenCV and write bytes in Python to avoid cv2.imwrite failures on Unicode paths."""
    ok, buf = cv2.imencode(".jpg", img)
    if not ok:
        return False
    try:
        with open(path, "wb") as f:
            f.write(buf.tobytes())
        return True
    except Exception:
        return False

# Upstream input: output directory from cut_video.py.
# It should contain minute_00, minute_01, ... and files like t_0s.jpg, t_10s.jpg.
# If no output appears, verify:
# 1) FRAMES_ROOT exists and points to the intended video
# 2) FRAMES_ROOT has minute_* folders with .jpg files
# macOS paths (change VIDEO_FOLDER for another class/video)
PROJECT_LLM_DIR = Path("/Users/alyssa/Desktop/llm_as_a_judge/data/llm")
VIDEO_FOLDER = "Zoom Meeting for 3D Design"
FRAMES_ROOT = str(PROJECT_LLM_DIR / "project" / VIDEO_FOLDER / "frames")
OUT_DIR = str(PROJECT_LLM_DIR / "project" / VIDEO_FOLDER / "images")
# Contour filtering thresholds (tune by image resolution if needed)
MIN_CONTOUR_AREA = 20000
MAX_CONTOUR_AREA = 500000
APPROXIMATION_FACTOR = 0.02
MEAN_INTENSITY_THRESHOLD = 50

os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.isdir(FRAMES_ROOT):
    raise RuntimeError(f"FRAMES_ROOT not found: {FRAMES_ROOT}")

# One folder per minute: minute_00, minute_01, ...
minute_dirs = sorted([
    d for d in os.listdir(FRAMES_ROOT)
    if os.path.isdir(os.path.join(FRAMES_ROOT, d)) and d.startswith("minute_")
])

print(f"FRAMES_ROOT: {FRAMES_ROOT}")
print(f"Found {len(minute_dirs)} minute_* folders: {minute_dirs[:5]}{'...' if len(minute_dirs) > 5 else ''}")

if not minute_dirs:
    raise RuntimeError(f"No minute_* folders found in {FRAMES_ROOT}")

total_written = 0


def detect_zoom_tiles(frame: np.ndarray):
    """
    Detect rectangular participant tiles from a Zoom-like screenshot using:
    1) grayscale + blur
    2) Canny edge detection
    3) contour extraction
    4) rectangular contour filtering + mean intensity filtering
    Returns list of (x1, y1, x2, y2) in reading order.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_CONTOUR_AREA < area < MAX_CONTOUR_AREA):
            continue

        epsilon = APPROXIMATION_FACTOR * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) != 4:
            continue
        if not cv2.isContourConvex(approx):
            continue

        x, y, bw, bh = cv2.boundingRect(approx)
        if bw <= 0 or bh <= 0:
            continue

        roi = gray[y:y + bh, x:x + bw]
        if roi.size == 0:
            continue
        mean_intensity = float(np.mean(roi))
        if mean_intensity <= MEAN_INTENSITY_THRESHOLD:
            continue

        boxes.append((x, y, x + bw, y + bh))

    if not boxes:
        return []

    # Sort by rows then columns to assign stable grid IDs.
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    heights = [b[3] - b[1] for b in boxes]
    row_tol = max(20, int(np.median(heights) * 0.35))

    rows = []
    for b in boxes:
        placed = False
        y_center = (b[1] + b[3]) // 2
        for row in rows:
            if abs(y_center - row["y_center"]) <= row_tol:
                row["boxes"].append(b)
                ys = [((rb[1] + rb[3]) // 2) for rb in row["boxes"]]
                row["y_center"] = int(sum(ys) / len(ys))
                placed = True
                break
        if not placed:
            rows.append({"y_center": y_center, "boxes": [b]})

    rows.sort(key=lambda r: r["y_center"])
    ordered = []
    for row in rows:
        ordered.extend(sorted(row["boxes"], key=lambda b: b[0]))
    return ordered


for minute_id in minute_dirs:
    minute_path = os.path.join(FRAMES_ROOT, minute_id)
    image_files = sorted([
        f for f in os.listdir(minute_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"[SKIP] {minute_id}: no images in {minute_path}")
        continue

    print(f"\nProcessing {minute_id}: {len(image_files)} images")

    # Output root for this minute: images_out/minute_00/, minute_01/, ...
    minute_out_base = os.path.join(OUT_DIR, minute_id)
    os.makedirs(minute_out_base, exist_ok=True)

    for img_name in image_files:
        img_path = os.path.join(minute_path, img_name)
        frame = read_image_unicode_safe(img_path)

        if frame is None:
            print(f"[WARN] Cannot read: {img_path}")
            continue

        if frame.ndim != 3:
            continue

        h, w, _ = frame.shape
        if h == 0 or w == 0:
            continue

        # Use filename as time tag (e.g. t_0s.jpg -> t_0s)
        time_tag = os.path.splitext(img_name)[0]

        boxes = detect_zoom_tiles(frame)
        if not boxes:
            print(f"[WARN] No tile boxes detected: {img_path}")
            continue

        for gid, (x1, y1, x2, y2) in enumerate(boxes):
            if y2 <= y1 or x2 <= x1:
                continue
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            grid_dir = os.path.join(minute_out_base, f"grid_{gid:02d}")
            os.makedirs(grid_dir, exist_ok=True)
            out_path = os.path.join(grid_dir, f"{time_tag}.jpg")
            if write_image_unicode_safe(out_path, crop):
                total_written += 1

print(f"\nFinished. Total images written: {total_written}")
print(f"Output directory: {OUT_DIR}")
