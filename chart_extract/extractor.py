from PIL import Image
import numpy as np
import sys

def extract_and_interpolate_trends(image_path: str):
    """
    Final version with all requested steps:
    - Detect chart bounding box
    - Extract topmost blue pixel per column
    - Remove the last value
    - Normalize (100 at top)
    - Shift minimum to 0
    - Interpolate to exactly 2048 elements
    """
    # Load images
    original = Image.open(image_path)
    gray = np.array(original.convert('L'), dtype=np.uint8)
    orig_h, orig_w = gray.shape
    print(f"Original image size: {orig_w}x{orig_h}")

    # Grid detection parameters
    THRESHOLD = 245
    MAX_THICKNESS = 5
    MIN_SPACING = 25
    MAX_SPACING = 45
    SPACING_TOL = 10
    Y_TOL = 6

    def get_thin_line_clusters(col):
        is_line = col < THRESHOLD
        clusters = []
        i = 0
        while i < orig_h:
            if is_line[i]:
                start = i
                while i < orig_h and is_line[i]:
                    i += 1
                thickness = i - start
                if 1 <= thickness <= MAX_THICKNESS:
                    center = (start + i - 1) // 2
                    clusters.append(center)
            else:
                i += 1
        return np.sort(np.array(clusters, dtype=int))

    # === 1. Find LEFT edge ===
    left_x = None
    y_centers = None
    for x in range(20, min(orig_w // 2, 400)):
        clusters = get_thin_line_clusters(gray[:, x])
        if len(clusters) >= 5:
            for i in range(len(clusters) - 4):
                sub = clusters[i:i+5]
                diffs = np.diff(sub)
                mean_spacing = float(np.mean(diffs))
                if (MIN_SPACING < mean_spacing < MAX_SPACING and
                    np.all(np.abs(diffs - mean_spacing) <= SPACING_TOL)):
                    left_x = x
                    y_centers = sub
                    print(f"✅ Left edge found at x = {left_x}")
                    print(f"   Grid y-centers: {y_centers.tolist()}")
                    break
        if left_x is not None:
            break

    if left_x is None or y_centers is None:
        raise ValueError("Could not find left chart boundary.")

    top = int(y_centers[0] - 3)
    bottom = int(y_centers[-1] + 3)

    # === 2. Find RIGHT edge ===
    right_x = left_x
    for x in range(left_x + 100, orig_w - 20):
        clusters = get_thin_line_clusters(gray[:, x])
        if len(clusters) >= 5:
            for i in range(len(clusters) - 4):
                sub = clusters[i:i+5]
                if np.max(np.abs(sub - y_centers)) <= Y_TOL:
                    right_x = x
                    break

    print(f"✅ Right edge found at x = {right_x}")

    # === 3. Crop chart ===
    bbox = (left_x, top, right_x, bottom)
    cropped = original.crop(bbox)
    cropped_array = np.array(cropped)
    height, width = cropped_array.shape[:2]
    print(f"Cropped chart size: {width}x{height}")

    # === 4. Extract topmost colored pixel per column ===
    timeseries_raw = []
    for x in range(width):
        column = cropped_array[:, x, :]
        r, g, b = column[:, 0], column[:, 1], column[:, 2]

        is_colored = (
            (np.abs(r.astype(int) - g.astype(int)) > 20) |
            (np.abs(r.astype(int) - b.astype(int)) > 20) |
            (np.abs(g.astype(int) - b.astype(int)) > 20)
        ) & (r < 200)

        colored_ys = np.where(is_colored)[0]
        if len(colored_ys) > 0:
            topmost_y = int(np.min(colored_ys))
            timeseries_raw.append(topmost_y)

    # === Remove the last value ===
    if len(timeseries_raw) > 0:
        timeseries_raw = timeseries_raw[:-1]
        print(f"   Removed last value. Raw length now: {len(timeseries_raw)}")

    # === 5. Normalize (100 at top) + shift minimum to 0 ===
    if timeseries_raw:
        normalized = [round(100 - (y / height * 100)) for y in timeseries_raw]
        
        min_val = min(normalized)
        final_timeseries = [x - min_val for x in normalized]

        print(f"   Min shifted to 0 (was {min_val})")

        # === 6. Interpolate to exactly 2048 elements ===
        if len(final_timeseries) > 1:
            x_old = np.linspace(0, 1, len(final_timeseries))
            x_new = np.linspace(0, 1, 2048)
            interpolated = np.interp(x_new, x_old, final_timeseries)
            final_timeseries = [round(float(x)) for x in interpolated]
            print(f"   Interpolated from {len(normalized)} to exactly 2048 values")
        else:
            print("Warning: Too few points for interpolation.")

        print(f"\n✅ Final interpolated timeseries with 2048 values")
        print("Final Timeseries:")
        print(final_timeseries)
        return final_timeseries
    else:
        print("No colored pixels found!")
        return []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_trends_timeseries.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        timeseries = extract_chart_bbox_and_timeseries(image_path)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
