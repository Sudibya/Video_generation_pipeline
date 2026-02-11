#!/usr/bin/env python3
"""
Image Comparator
Find and report differences between two images
"""

import cv2
import numpy as np


def detect_crop(image, image_name):
    """
    Detect if an image has been cropped (has gray/blank fill from corruption)

    Only detects ACTUAL gray fill (corrupted JPEG, solid color blocks).
    Does NOT flag natural low-contrast areas like sky, haze, or shadows.

    Detection criteria (ALL must be true for a row to count as blank):
    1. Row pixel range < 10 (nearly solid color, not just low contrast)
    2. Row is nearly identical to its neighbor (diff < 2)
    3. R, G, B channels are almost equal (actual gray, not tinted)

    Args:
        image: Loaded image (numpy array)
        image_name: Name of the image for display

    Returns:
        Dictionary with crop info for each side
    """

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)

    crop_info = {
        'top': 0,
        'bottom': 0,
        'left': 0,
        'right': 0,
        'is_cropped': False
    }

    def is_blank_row(row_idx, neighbor_idx):
        """Check if a row is truly blank/gray fill (not just low contrast)"""
        row_pixels = gray[row_idx, :]

        # 1. Nearly solid color (pixel range < 10)
        pixel_range = int(np.max(row_pixels)) - int(np.min(row_pixels))
        if pixel_range >= 10:
            return False

        # 2. Nearly identical to neighbor row (diff < 2)
        neighbor_pixels = gray[neighbor_idx, :]
        row_diff = np.mean(np.abs(row_pixels.astype(float) - neighbor_pixels.astype(float)))
        if row_diff >= 2:
            return False

        # 3. R, G, B channels are almost equal (true gray, not tinted sky)
        r_mean = float(np.mean(r[row_idx, :]))
        g_mean = float(np.mean(g[row_idx, :]))
        b_mean = float(np.mean(b[row_idx, :]))
        max_channel_diff = max(abs(r_mean - g_mean), abs(r_mean - b_mean), abs(g_mean - b_mean))
        if max_channel_diff >= 5:
            return False

        return True

    # Scan BOTTOM
    for row in range(height - 1, 0, -1):
        if is_blank_row(row, row - 1):
            crop_info['bottom'] += 1
        else:
            break

    # Scan TOP
    for row in range(0, height - 1):
        if is_blank_row(row, row + 1):
            crop_info['top'] += 1
        else:
            break

    # Scan LEFT
    # (Use column checks - transpose logic)
    for col in range(0, width - 1):
        col_pixels = gray[:, col]
        next_pixels = gray[:, col + 1]
        pixel_range = int(np.max(col_pixels)) - int(np.min(col_pixels))
        col_diff = np.mean(np.abs(col_pixels.astype(float) - next_pixels.astype(float)))
        if pixel_range < 10 and col_diff < 2:
            crop_info['left'] += 1
        else:
            break

    # Scan RIGHT
    for col in range(width - 1, 0, -1):
        col_pixels = gray[:, col]
        prev_pixels = gray[:, col - 1]
        pixel_range = int(np.max(col_pixels)) - int(np.min(col_pixels))
        col_diff = np.mean(np.abs(col_pixels.astype(float) - prev_pixels.astype(float)))
        if pixel_range < 10 and col_diff < 2:
            crop_info['right'] += 1
        else:
            break

    # Mark as cropped only if significant blank area found (> 100px)
    min_crop_pixels = 100
    crop_info['is_cropped'] = any([
        crop_info['top'] > min_crop_pixels,
        crop_info['bottom'] > min_crop_pixels,
        crop_info['left'] > min_crop_pixels,
        crop_info['right'] > min_crop_pixels,
    ])

    # Print crop detection results
    print(f"\n   --- Crop Detection for {image_name} ---")

    if crop_info['is_cropped']:
        print(f"   ✗ IMAGE IS CROPPED! Missing content detected:")
        if crop_info['top'] > min_crop_pixels:
            pct = (crop_info['top'] / height) * 100
            print(f"     TOP:    {crop_info['top']}px blank ({pct:.1f}% of height)")
        if crop_info['bottom'] > min_crop_pixels:
            pct = (crop_info['bottom'] / height) * 100
            print(f"     BOTTOM: {crop_info['bottom']}px blank ({pct:.1f}% of height)")
        if crop_info['left'] > min_crop_pixels:
            pct = (crop_info['left'] / width) * 100
            print(f"     LEFT:   {crop_info['left']}px blank ({pct:.1f}% of width)")
        if crop_info['right'] > min_crop_pixels:
            pct = (crop_info['right'] / width) * 100
            print(f"     RIGHT:  {crop_info['right']}px blank ({pct:.1f}% of width)")

        actual_w = width - crop_info['left'] - crop_info['right']
        actual_h = height - crop_info['top'] - crop_info['bottom']
        print(f"     Actual content area: {actual_w}x{actual_h} pixels")
        print(f"     Total image size:    {width}x{height} pixels")
    else:
        print(f"   ✓ Image is NOT cropped (full content)")

    return crop_info


def detect_haze(image, image_name):
    """
    Detect haze/cloudiness in an image using the Dark Channel Prior.

    How it works:
    - For each pixel, take the MINIMUM value across R, G, B channels
    - Then apply a minimum filter (erosion) over a small patch
    - In a haze-FREE image, most patches have at least one very dark pixel
      (shadows, dark objects, etc.), so the dark channel is close to 0
    - In a HAZY image, haze adds a white veil everywhere, so even the
      darkest pixels are lifted up, making the dark channel bright

    Returns:
        dict with:
        - dark_channel_mean: average dark channel intensity (0-255)
        - haze_density: percentage (0-100%)
        - is_hazy: True if significant haze detected
        - level: "clear", "slight", "moderate", "heavy"
    """

    # Step 1: Compute dark channel
    # For each pixel, take the minimum across B, G, R
    min_channel = np.min(image, axis=2)

    # Step 2: Apply minimum filter (erosion) with a patch
    # Patch size 15x15 works well for typical images
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # Step 3: Measure haze level
    dc_mean = float(np.mean(dark_channel))
    dc_median = float(np.median(dark_channel))

    # Percentage of pixels with dark channel > 50 (haze-affected pixels)
    hazy_pixels = np.sum(dark_channel > 50)
    total_pixels = dark_channel.shape[0] * dark_channel.shape[1]
    haze_density = (hazy_pixels / total_pixels) * 100

    # Classify haze level
    # dc_mean < 30: clear
    # dc_mean 30-60: slight haze
    # dc_mean 60-100: moderate haze
    # dc_mean > 100: heavy haze
    if dc_mean < 30:
        level = "clear"
        is_hazy = False
    elif dc_mean < 60:
        level = "slight"
        is_hazy = True
    elif dc_mean < 100:
        level = "moderate"
        is_hazy = True
    else:
        level = "heavy"
        is_hazy = True

    # Print results
    print(f"\n   --- Haze Detection for {image_name} ---")
    if is_hazy:
        print(f"   ✗ HAZE DETECTED! Level: {level.upper()}")
        print(f"     Dark channel mean: {dc_mean:.1f}/255 (lower = clearer)")
        print(f"     Haze density: {haze_density:.1f}% of pixels affected")
        if level == "heavy":
            print(f"     → Heavy cloudiness. Will look washed out in video.")
        elif level == "moderate":
            print(f"     → Noticeable haze. Colors will appear muted.")
        else:
            print(f"     → Mild haze. Slight loss of contrast.")
    else:
        print(f"   ✓ Image is CLEAR (no significant haze)")
        print(f"     Dark channel mean: {dc_mean:.1f}/255")

    return {
        'dark_channel_mean': dc_mean,
        'dark_channel_median': dc_median,
        'haze_density': haze_density,
        'is_hazy': is_hazy,
        'level': level
    }


def analyze_image_quality(image, image_name):
    """
    Analyze brightness, contrast, color balance, sharpness, and exposure

    Args:
        image: Loaded image (numpy array)
        image_name: Name for display

    Returns:
        Dictionary with all quality metrics
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]

    # Split into B, G, R channels
    b, g, r = cv2.split(image)

    # 1. BRIGHTNESS (average pixel intensity 0-255)
    brightness = np.mean(gray)

    # 2. CONTRAST (standard deviation of pixel intensity)
    contrast = np.std(gray)

    # 3. COLOR BALANCE (average of each channel)
    avg_blue = np.mean(b)
    avg_green = np.mean(g)
    avg_red = np.mean(r)

    # Detect color shift (which channel dominates)
    color_diff_rg = abs(avg_red - avg_green)
    color_diff_rb = abs(avg_red - avg_blue)
    color_diff_gb = abs(avg_green - avg_blue)

    # 4. SHARPNESS (Laplacian variance - higher = sharper)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = laplacian.var()

    # 5. EXPOSURE
    # Count overexposed pixels (very bright, > 240)
    overexposed = np.sum(gray > 240)
    overexposed_pct = (overexposed / (height * width)) * 100

    # Count underexposed pixels (very dark, < 15)
    underexposed = np.sum(gray < 15)
    underexposed_pct = (underexposed / (height * width)) * 100

    # 6. HAZE (Dark Channel Prior)
    min_channel = np.min(image, axis=2)
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    haze_dc_mean = float(np.mean(dark_channel))

    if haze_dc_mean < 30:
        haze_level = "clear"
        is_hazy = False
    elif haze_dc_mean < 60:
        haze_level = "slight"
        is_hazy = True
    elif haze_dc_mean < 100:
        haze_level = "moderate"
        is_hazy = True
    else:
        haze_level = "heavy"
        is_hazy = True

    # 7. COLOR DEPTH / VIBRANCY
    # Measures how "deep" or "washed out" colors appear
    #
    # Local contrast: Average standard deviation in small patches (32x32)
    #   High = colors have depth, shadows are dark, highlights are bright
    #   Low  = flat, washed out, everything looks similar
    #
    # Saturation spread: Standard deviation of saturation channel
    #   High = mix of vivid and muted areas (natural, rich look)
    #   Low  = everything same saturation (flat, dull look)
    #
    # Vivid pixel %: Percentage of pixels with saturation > 100
    #   High = lots of deep/vivid colors
    #   Low  = mostly muted/light colors

    # Local contrast (compute std dev in 32x32 patches)
    patch = 32
    local_stds = []
    for y in range(0, height - patch, patch):
        for x in range(0, width - patch, patch):
            tile = gray[y:y+patch, x:x+patch]
            local_stds.append(np.std(tile))
    local_contrast = float(np.mean(local_stds))

    # Saturation analysis
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_channel = hsv[:, :, 1].astype(float)
    sat_mean = float(np.mean(sat_channel))
    sat_std = float(np.std(sat_channel))
    vivid_pixels = float(np.sum(sat_channel > 100) / (height * width) * 100)

    # Overall color depth score (0-100)
    # Combines local contrast, saturation spread, and vivid pixel %
    depth_score = (local_contrast / 80 * 33) + (sat_std / 60 * 33) + (vivid_pixels / 50 * 34)
    depth_score = min(depth_score, 100.0)

    if depth_score > 60:
        depth_level = "deep"
    elif depth_score > 35:
        depth_level = "moderate"
    else:
        depth_level = "light"

    quality = {
        'brightness': brightness,
        'contrast': contrast,
        'avg_red': avg_red,
        'avg_green': avg_green,
        'avg_blue': avg_blue,
        'sharpness': sharpness,
        'overexposed_pct': overexposed_pct,
        'underexposed_pct': underexposed_pct,
        'haze_dc_mean': haze_dc_mean,
        'haze_level': haze_level,
        'is_hazy': is_hazy,
        'local_contrast': local_contrast,
        'sat_mean': sat_mean,
        'sat_std': sat_std,
        'vivid_pct': vivid_pixels,
        'depth_score': depth_score,
        'depth_level': depth_level,
    }

    return quality


def compare_image_quality(quality1, quality2, name1, name2):
    """
    Compare quality metrics between two images and print detailed report

    Args:
        quality1: Quality dict from image 1
        quality2: Quality dict from image 2
        name1: Image 1 name
        name2: Image 2 name
    """

    print(f"\n{'='*70}")
    print("IMAGE QUALITY COMPARISON:")
    print(f"{'='*70}")

    # --- BRIGHTNESS ---
    b1 = quality1['brightness']
    b2 = quality2['brightness']
    b_diff = b2 - b1

    print(f"\n   --- BRIGHTNESS (0=black, 255=white) ---")
    print(f"   Image 1: {b1:.1f}/255")
    print(f"   Image 2: {b2:.1f}/255")
    print(f"   Difference: {b_diff:+.1f}")

    if abs(b_diff) < 5:
        print(f"   ✓ Brightness is CONSISTENT")
    elif abs(b_diff) < 15:
        print(f"   ⚠ SLIGHT brightness difference (may cause minor flicker)")
    else:
        bright_img = "Image 2" if b_diff > 0 else "Image 1"
        print(f"   ✗ SIGNIFICANT brightness difference! ({bright_img} is brighter)")
        print(f"     → Will cause FLICKER in video. Needs correction.")

    # --- CONTRAST ---
    c1 = quality1['contrast']
    c2 = quality2['contrast']
    c_diff = c2 - c1

    print(f"\n   --- CONTRAST (higher = more contrast) ---")
    print(f"   Image 1: {c1:.1f}")
    print(f"   Image 2: {c2:.1f}")
    print(f"   Difference: {c_diff:+.1f}")

    if abs(c_diff) < 5:
        print(f"   ✓ Contrast is CONSISTENT")
    elif abs(c_diff) < 15:
        print(f"   ⚠ SLIGHT contrast difference")
    else:
        high_img = "Image 2" if c_diff > 0 else "Image 1"
        print(f"   ✗ SIGNIFICANT contrast difference! ({high_img} has more contrast)")
        print(f"     → Will cause VISUAL JUMP in video. Needs correction.")

    # --- COLOR BALANCE ---
    print(f"\n   --- COLOR BALANCE (R/G/B channel averages) ---")
    print(f"   Image 1: R={quality1['avg_red']:.1f}  G={quality1['avg_green']:.1f}  B={quality1['avg_blue']:.1f}")
    print(f"   Image 2: R={quality2['avg_red']:.1f}  G={quality2['avg_green']:.1f}  B={quality2['avg_blue']:.1f}")

    r_diff = quality2['avg_red'] - quality1['avg_red']
    g_diff = quality2['avg_green'] - quality1['avg_green']
    b_diff_ch = quality2['avg_blue'] - quality1['avg_blue']

    print(f"   Shift:    R={r_diff:+.1f}      G={g_diff:+.1f}      B={b_diff_ch:+.1f}")

    max_shift = max(abs(r_diff), abs(g_diff), abs(b_diff_ch))
    if max_shift < 5:
        print(f"   ✓ Color balance is CONSISTENT")
    elif max_shift < 15:
        print(f"   ⚠ SLIGHT color shift detected")
    else:
        shifts = []
        if abs(r_diff) > 10:
            shifts.append(f"Red {'↑' if r_diff > 0 else '↓'}")
        if abs(g_diff) > 10:
            shifts.append(f"Green {'↑' if g_diff > 0 else '↓'}")
        if abs(b_diff_ch) > 10:
            shifts.append(f"Blue {'↑' if b_diff_ch > 0 else '↓'}")
        print(f"   ✗ SIGNIFICANT color shift! ({', '.join(shifts)} in Image 2)")
        print(f"     → Will cause COLOR FLICKER in video. Needs correction.")

    # --- SHARPNESS ---
    s1 = quality1['sharpness']
    s2 = quality2['sharpness']
    s_ratio = s2 / s1 if s1 > 0 else 0

    print(f"\n   --- SHARPNESS (higher = sharper) ---")
    print(f"   Image 1: {s1:.1f}")
    print(f"   Image 2: {s2:.1f}")
    print(f"   Ratio: {s_ratio:.2f}x")

    if 0.8 < s_ratio < 1.2:
        print(f"   ✓ Sharpness is CONSISTENT")
    elif 0.5 < s_ratio < 1.5:
        blurry_img = "Image 2" if s_ratio < 1 else "Image 1"
        print(f"   ⚠ SLIGHT sharpness difference ({blurry_img} is blurrier)")
    else:
        blurry_img = "Image 2" if s_ratio < 1 else "Image 1"
        print(f"   ✗ SIGNIFICANT sharpness difference! ({blurry_img} is much blurrier)")
        print(f"     → Will cause BLUR FLICKER in video.")

    # --- EXPOSURE ---
    print(f"\n   --- EXPOSURE ---")
    print(f"   Image 1: Overexposed={quality1['overexposed_pct']:.1f}%  Underexposed={quality1['underexposed_pct']:.1f}%")
    print(f"   Image 2: Overexposed={quality2['overexposed_pct']:.1f}%  Underexposed={quality2['underexposed_pct']:.1f}%")

    over_diff = abs(quality2['overexposed_pct'] - quality1['overexposed_pct'])
    under_diff = abs(quality2['underexposed_pct'] - quality1['underexposed_pct'])

    if over_diff < 3 and under_diff < 3:
        print(f"   ✓ Exposure is CONSISTENT")
    else:
        if over_diff >= 3:
            over_img = "Image 2" if quality2['overexposed_pct'] > quality1['overexposed_pct'] else "Image 1"
            print(f"   ⚠ {over_img} has more OVEREXPOSED (blown out) areas")
        if under_diff >= 3:
            under_img = "Image 2" if quality2['underexposed_pct'] > quality1['underexposed_pct'] else "Image 1"
            print(f"   ⚠ {under_img} has more UNDEREXPOSED (too dark) areas")

    # --- HAZE ---
    print(f"\n   --- HAZE / CLOUDINESS ---")
    h1_dc = quality1['haze_dc_mean']
    h2_dc = quality2['haze_dc_mean']
    h1_level = quality1['haze_level']
    h2_level = quality2['haze_level']

    print(f"   Image 1: {h1_level.upper()} (dark channel: {h1_dc:.1f})")
    print(f"   Image 2: {h2_level.upper()} (dark channel: {h2_dc:.1f})")

    if quality1['is_hazy'] == quality2['is_hazy'] and abs(h1_dc - h2_dc) < 15:
        print(f"   ✓ Haze level is CONSISTENT")
    else:
        hazier = "Image 1" if h1_dc > h2_dc else "Image 2"
        clearer = "Image 2" if h1_dc > h2_dc else "Image 1"
        print(f"   ✗ HAZE MISMATCH! {hazier} is hazier than {clearer}")
        print(f"     Difference: {abs(h1_dc - h2_dc):.1f} dark channel units")
        print(f"     → Will cause CLARITY FLICKER in video. Needs dehazing.")

    # --- COLOR DEPTH / VIBRANCY ---
    print(f"\n   --- COLOR DEPTH / VIBRANCY ---")

    d1 = quality1['depth_score']
    d2 = quality2['depth_score']
    dl1 = quality1['depth_level']
    dl2 = quality2['depth_level']

    print(f"   Image 1: {dl1.upper()} (score: {d1:.1f}/100)")
    print(f"     Local contrast: {quality1['local_contrast']:.1f}")
    print(f"     Saturation spread: {quality1['sat_std']:.1f}")
    print(f"     Vivid pixels: {quality1['vivid_pct']:.1f}%")

    print(f"   Image 2: {dl2.upper()} (score: {d2:.1f}/100)")
    print(f"     Local contrast: {quality2['local_contrast']:.1f}")
    print(f"     Saturation spread: {quality2['sat_std']:.1f}")
    print(f"     Vivid pixels: {quality2['vivid_pct']:.1f}%")

    depth_diff = abs(d1 - d2)
    if depth_diff < 5:
        print(f"   ✓ Color depth is CONSISTENT")
    elif depth_diff < 15:
        lighter = "Image 2" if d2 < d1 else "Image 1"
        deeper = "Image 1" if d2 < d1 else "Image 2"
        print(f"   ⚠ SLIGHT color depth difference ({lighter} has lighter colors)")
    else:
        lighter = "Image 2" if d2 < d1 else "Image 1"
        deeper = "Image 1" if d2 < d1 else "Image 2"
        print(f"   ✗ SIGNIFICANT color depth difference!")
        print(f"     {deeper} has DEEPER colors (score: {max(d1,d2):.1f})")
        print(f"     {lighter} has LIGHTER colors (score: {min(d1,d2):.1f})")
        print(f"     → Will cause COLOR DEPTH FLICKER in video. Needs local contrast enhancement.")


def compare_images(image1_path, image2_path):
    """
    Compare two images and report differences

    Args:
        image1_path: Path to first image (reference)
        image2_path: Path to second image (to compare)
    """

    print("\n" + "="*70)
    print("COMPARING IMAGES")
    print("="*70)

    # Load images
    print(f"\nLoading images...")
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: Could not load one or both images")
        return None

    # Get dimensions
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    print(f"\nImage 1 ({image1_path}): {w1}x{h1} pixels")
    print(f"Image 2 ({image2_path}): {w2}x{h2} pixels")

    # Detect if either image is cropped
    print(f"\n{'='*70}")
    print("CROP DETECTION:")
    print(f"{'='*70}")
    crop1 = detect_crop(image1, f"Image 1 ({image1_path})")
    crop2 = detect_crop(image2, f"Image 2 ({image2_path})")

    if crop1['is_cropped'] or crop2['is_cropped']:
        print(f"\n   ⚠ WARNING: Cropped image(s) detected!")
        print(f"   → Run image_aligner.py to fix before video generation")

    # Detect haze/cloudiness
    print(f"\n{'='*70}")
    print("HAZE DETECTION:")
    print(f"{'='*70}")
    haze1 = detect_haze(image1, f"Image 1 ({image1_path})")
    haze2 = detect_haze(image2, f"Image 2 ({image2_path})")

    if haze1['is_hazy'] or haze2['is_hazy']:
        if haze1['is_hazy'] != haze2['is_hazy']:
            hazier = "Image 1" if haze1['dark_channel_mean'] > haze2['dark_channel_mean'] else "Image 2"
            print(f"\n   ⚠ WARNING: {hazier} is hazy while the other is clear!")
            print(f"   → Run image_corrector.py to dehaze before video generation")

    # Analyze image quality (brightness, contrast, color, sharpness, exposure, haze)
    quality1 = analyze_image_quality(image1, image1_path)
    quality2 = analyze_image_quality(image2, image2_path)
    compare_image_quality(quality1, quality2, image1_path, image2_path)

    # Convert to grayscale for feature detection
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Detect features
    print(f"\nDetecting features...")
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, desc1 = orb.detectAndCompute(gray1, None)
    kp2, desc2 = orb.detectAndCompute(gray2, None)

    print(f"Image 1: {len(kp1)} features detected")
    print(f"Image 2: {len(kp2)} features detected")

    # Match features
    print(f"\nMatching features between images...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"Found {len(matches)} matching features\n")

    # Print detailed differences
    print("="*70)
    print("DIFFERENCES FOUND:")
    print("="*70)

    print(f"\n1. SIZE DIFFERENCE:")
    print(f"   Width:  {w1} vs {w2} (difference: {w1-w2:+d}px)")
    print(f"   Height: {h1} vs {h2} (difference: {h1-h2:+d}px)")

    print(f"\n2. FEATURE COUNT:")
    print(f"   Image 1: {len(kp1)} features")
    print(f"   Image 2: {len(kp2)} features")
    print(f"   Difference: {abs(len(kp1)-len(kp2)):+d} features")

    print(f"\n3. FEATURE MATCHING:")
    print(f"   Matching features: {len(matches)}")
    print(f"   Match coverage: {(len(matches)/max(len(kp1), len(kp2))*100):.1f}% matched")

    if len(matches) > 0:
        print(f"\n4. MATCH QUALITY:")
        distances = [m.distance for m in matches]
        print(f"   Best match quality: {min(distances)}/256 (lower is better)")
        print(f"   Worst match quality: {max(distances)}/256")
        print(f"   Average quality: {sum(distances)/len(distances):.1f}/256")

    # Show first few matching keypoints
    if len(matches) > 0:
        print(f"\n5. SAMPLE MATCHING KEYPOINTS (first 5):")
        for i in range(min(5, len(matches))):
            m = matches[i]
            pt1 = kp1[m.queryIdx].pt
            pt2 = kp2[m.trainIdx].pt
            print(f"\n   Match {i+1}:")
            print(f"     Image 1 position: ({pt1[0]:.1f}, {pt1[1]:.1f})")
            print(f"     Image 2 position: ({pt2[0]:.1f}, {pt2[1]:.1f})")
            print(f"     Position shift: dx={pt1[0]-pt2[0]:+.1f}px, dy={pt1[1]-pt2[1]:+.1f}px")
            print(f"     Match quality: {m.distance}/256")

    print(f"\n" + "="*70)
    print("SUMMARY:")
    print("="*70)

    # Determine alignment status
    if len(matches) > 0:
        avg_dx = np.mean([kp1[m.queryIdx].pt[0] - kp2[m.trainIdx].pt[0] for m in matches])
        avg_dy = np.mean([kp1[m.queryIdx].pt[1] - kp2[m.trainIdx].pt[1] for m in matches])

        print(f"\nAverage position shift:")
        print(f"  Horizontal (X): {avg_dx:+.2f}px")
        print(f"  Vertical (Y):   {avg_dy:+.2f}px")

        if abs(avg_dx) < 5 and abs(avg_dy) < 5:
            print(f"\n✓ Images are WELL-ALIGNED (minimal shift)")
        elif abs(avg_dx) < 20 and abs(avg_dy) < 20:
            print(f"\n⚠ Images have SMALL misalignment (may cause slight jitter)")
        else:
            print(f"\n✗ Images are MISALIGNED (will cause visible shaking in video)")

    print(f"\n" + "="*70)

    return {
        'image1': image1,
        'image2': image2,
        'kp1': kp1,
        'kp2': kp2,
        'matches': matches,
        'desc1': desc1,
        'desc2': desc2,
        'crop1': crop1,
        'crop2': crop2,
        'haze1': haze1,
        'haze2': haze2,
        'quality1': quality1,
        'quality2': quality2
    }


def main():
    """Run comparison on images - accepts command line arguments"""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 image_comparator.py <image1_path> <image2_path>")
        print("\nExample:")
        print("  python3 image_comparator.py image.png testing.jpg")
        print("  python3 image_comparator.py 20250608143005.JPG 20250608150010.JPG")
        sys.exit(1)

    image1_path = sys.argv[1]
    image2_path = sys.argv[2]

    result = compare_images(image1_path, image2_path)
    return result


if __name__ == "__main__":
    main()
