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

    quality = {
        'brightness': brightness,
        'contrast': contrast,
        'avg_red': avg_red,
        'avg_green': avg_green,
        'avg_blue': avg_blue,
        'sharpness': sharpness,
        'overexposed_pct': overexposed_pct,
        'underexposed_pct': underexposed_pct,
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

    # Analyze image quality (brightness, contrast, color, sharpness, exposure)
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
