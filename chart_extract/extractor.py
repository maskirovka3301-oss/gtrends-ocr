import cv2
import numpy as np
import easyocr
import json
import os
import sys
from scipy import interpolate

def find_grid_line_y(img, text_coords, search_x_start_offset=20):
    """
    Finds the y-coordinate of the horizontal grid line associated with a text marker.
    Handles interrupted/dashed lines by looking for the total span and density of gray pixels.
    """
    if not text_coords:
        return None, None, None
    
    x, y, w, h = text_coords
    search_x_start = x + w + search_x_start_offset
    height, width, _ = img.shape
    
    y_scan_start = max(0, y - 20)
    y_scan_end = min(height, y + h + 20)
    
    best_row = None
    max_gray_pixels = 0
    best_start_x = None
    best_span = None
    
    for row_y in range(y_scan_start, y_scan_end):
        current_gray_count = 0
        current_min_x = -1
        current_max_x = -1
        
        for col_x in range(search_x_start, width):
            pixel = img[row_y, col_x]
            # Check for gray (Google Trends grid lines are light gray)
            if abs(int(pixel[0]) - int(pixel[1])) < 15 and \
               abs(int(pixel[1]) - int(pixel[2])) < 15 and \
               220 < int(pixel[0]) < 250: 
                
                current_gray_count += 1
                if current_min_x == -1:
                    current_min_x = col_x
                current_max_x = col_x
        
        # Threshold: A line should have a reasonable number of pixels
        if current_gray_count > 50: 
            # Prefer the row with the most pixels (longest line)
            if current_gray_count > max_gray_pixels:
                max_gray_pixels = current_gray_count
                best_row = row_y
                best_start_x = current_min_x
                # Span is the distance from the first pixel to the last pixel
                # This handles dashed lines correctly by including the gaps in the width
                best_span = current_max_x - current_min_x

    if best_row is None:
        return None, None, None
        
    return best_row, best_span, best_start_x

def extract_and_interpolate_trends(image_path, output_points=1024):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Error: Could not load image {image_path}")

    height, width, _ = img.shape

    # --- EasyOCR Initialization ---
    # gpu=True enables hardware acceleration (CUDA, MPS, ROCm) if available.
    # Reader downloads models on first run.
    reader = easyocr.Reader(['en'], gpu=True)
    
    # Run OCR once for the whole image
    # Returns list of: (bbox, text, confidence)
    # bbox is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    results = reader.readtext(img)

    def find_text_in_results(text_target, min_y=0):
        """
        Searches the OCR results for a specific text string below a minimum Y coordinate.
        """
        for (bbox, text, conf) in results:
            # Clean text for comparison (remove trailing periods, whitespace)
            clean_text = text.strip().rstrip('.').lower()
            target_clean = text_target.lower()
            
            if clean_text == target_clean and conf > 0.5:
                # bbox[0] is top-left [x, y]
                x, y = bbox[0]
                # Calculate width and height from bbox corners
                w = bbox[1][0] - x
                h = bbox[2][1] - y
                
                if y >= min_y:
                    return (x, y, w, h)
        return None

    # 1. Locate "Interest over time" header first
    # We search from y=0
    interest_header_coords = find_text_in_results("Interest over time", min_y=0)
    
    if not interest_header_coords:
        raise ValueError("Error: Could not find 'Interest over time' header.")

    header_x, header_y, header_w, header_h = interest_header_coords
    
    # Define the search area for markers: strictly below the header
    # Add a small buffer to ensure we are past the title text
    chart_search_min_y = header_y + header_h + 50 

    # 2. Locate Markers (100, 75, 50, 25) ONLY below the header
    markers = ['100', '75', '50', '25']
    marker_coords = {}
    
    for m in markers:
        # Pass min_y to restrict search area
        coords = find_text_in_results(m, min_y=chart_search_min_y)
        if coords:
            marker_coords[m] = coords
        else:
            raise ValueError(f"Error: Could not find marker '{m}' below the chart header.")

    # 3. Locate Grid Lines
    grid_lines = {}
    for m in markers:
        if m in marker_coords:
            y, length, start_x = find_grid_line_y(img, marker_coords[m])
            if y is not None:
                grid_lines[m] = {'y': y, 'length': length, 'start_x': start_x}

    # Validate we have enough lines
    required_markers = ['100', '75', '50', '25']
    if not all(k in grid_lines for k in required_markers):
        raise ValueError("Error: Could not detect all necessary grid lines.")

    # 4. Calculate Dimensions
    y_100 = grid_lines['100']['y']
    y_75 = grid_lines['75']['y']
    y_50 = grid_lines['50']['y']
    y_25 = grid_lines['25']['y']
    
    # Calculate pixel distance for 25 units
    # Note: Y increases downwards, so higher values have smaller Y coordinates.
    dist_25_50 = y_50 - y_25
    dist_50_75 = y_75 - y_50
    dist_75_100 = y_100 - y_75
    
    # Average unit distance to be robust
    unit_dist = (dist_25_50 + dist_50_75 + dist_75_100) / 3.0
    
    # Calculate 0 position (25 units below 25)
    y_0 = y_25 + unit_dist
    
    # X-axis calculation
    start_x = min([grid_lines[k]['start_x'] for k in required_markers])
    max_len = max([grid_lines[k]['length'] for k in required_markers])
    end_x = start_x + max_len
    
    chart_x1, chart_y1 = start_x, y_100
    chart_x2, chart_y2 = end_x, int(y_0)
    
    # 5. Crop the chart
    chart_x1 = max(0, chart_x1)
    chart_y1 = max(0, chart_y1)
    chart_x2 = min(width, chart_x2)
    chart_y2 = min(height, chart_y2)
    
    chart_crop = img[chart_y1:chart_y2, chart_x1:chart_x2]
    
    # 6. Scan for data points (Blue line)
    hsv_crop = cv2.cvtColor(chart_crop, cv2.COLOR_BGR2HSV)
    
    # Google Trends blue is roughly #4285F4
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    mask = cv2.inRange(hsv_crop, lower_blue, upper_blue)
    
    time_series_raw = []
    chart_h, chart_w = mask.shape
    
    for x in range(chart_w):
        col_pixels = mask[:, x]
        indices = np.where(col_pixels > 0)[0]
        
        if len(indices) > 0:
            avg_y = np.mean(indices)
            # Normalize: y=0 (top of crop) is 100 value. y=chart_h (bottom) is 0 value.
            value = 100.0 - (avg_y / float(chart_h)) * 100.0
            time_series_raw.append(value)
        else:
            time_series_raw.append(None)

    # 7. Interpolate
    x_data = [i for i, v in enumerate(time_series_raw) if v is not None]
    y_data = [v for v in time_series_raw if v is not None]
    
    if len(x_data) < 2:
        raise ValueError("Error: Not enough data points found in chart.")

    # Cubic interpolation provides a smooth curve
    f = interpolate.interp1d(x_data, y_data, kind='cubic', fill_value="extrapolate")
    
    x_new = np.linspace(0, chart_w - 1, output_points)
    y_new = f(x_new)
    
    # Clamp values
    y_new = np.clip(y_new, 0, 100)
    
    final_series = y_new.tolist()

    # 8. Output JSON
    base_name = os.path.splitext(image_path)[0]
    output_json_path = f"{base_name}.json"
    
    with open(output_json_path, 'w') as f:
        json.dump(final_series, f, indent=4)
        
    print(f"Success: Extracted {len(final_series)} points to {output_json_path}")
    return final_series

if __name__ == "__main__":
    if len(sys.argv) > 1:
        extract_and_interpolate_trends(sys.argv[1])
    else:
        print("Usage: python extractor.py <image_path>")