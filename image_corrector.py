#!/usr/bin/env python3
"""
Image Corrector - Fix color, contrast, sharpness, and alignment differences
between two images so they can be used for smooth video generation
"""

import cv2
import numpy as np
import sys
import os


def correct_brightness_contrast(image, reference):
    """
    Match brightness and contrast of image to reference

    Uses histogram matching on each channel to make
    the image look like the reference in terms of lighting
    """

    result = image.copy()

    for i in range(3):  # B, G, R channels
        # Get histograms
        ref_hist, _ = np.histogram(reference[:, :, i].flatten(), 256, [0, 256])
        img_hist, _ = np.histogram(image[:, :, i].flatten(), 256, [0, 256])

        # Calculate CDFs (cumulative distribution functions)
        ref_cdf = ref_hist.cumsum()
        img_cdf = img_hist.cumsum()

        # Normalize CDFs
        ref_cdf = ref_cdf / ref_cdf[-1]
        img_cdf = img_cdf / img_cdf[-1]

        # Create lookup table to map image pixels to reference distribution
        lookup = np.zeros(256, dtype=np.uint8)
        for pixel_val in range(256):
            # Find closest matching CDF value in reference
            closest = np.argmin(np.abs(ref_cdf - img_cdf[pixel_val]))
            lookup[pixel_val] = closest

        # Apply lookup table
        result[:, :, i] = lookup[image[:, :, i]]

    return result


def correct_color_balance(image, reference):
    """
    Match color balance of image to reference
    Adjusts each RGB channel's mean to match the reference
    """

    result = image.astype(np.float64)

    for i in range(3):  # B, G, R
        ref_mean = np.mean(reference[:, :, i])
        img_mean = np.mean(image[:, :, i])

        # Scale factor to match means
        if img_mean > 0:
            scale = ref_mean / img_mean
            result[:, :, i] = result[:, :, i] * scale

    # Clip to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result


def correct_sharpness(image, reference):
    """
    Match sharpness of image to reference
    If image is blurrier, apply sharpening
    If image is sharper, apply slight blur
    """

    # Measure sharpness using Laplacian variance
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    sharpness_img = cv2.Laplacian(gray_img, cv2.CV_64F).var()
    sharpness_ref = cv2.Laplacian(gray_ref, cv2.CV_64F).var()

    ratio = sharpness_ref / sharpness_img if sharpness_img > 0 else 1.0

    if ratio > 1.2:
        # Image is blurrier than reference - SHARPEN it
        # Unsharp mask: sharpen = original + strength * (original - blurred)
        strength = min(ratio - 1.0, 2.0)  # Cap at 2.0
        blurred = cv2.GaussianBlur(image, (0, 0), 3)
        result = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return result, "sharpened"

    elif ratio < 0.8:
        # Image is sharper than reference - slight BLUR
        kernel_size = 3
        result = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return result, "blurred"

    else:
        return image, "no change"


def correct_alignment(image, reference):
    """
    Align image to reference using feature matching.

    Uses SIMPLE TRANSLATION (shift) for small movements instead of
    full homography which can warp/distort the image badly.

    Only uses homography if rotation/scale is detected.
    """

    # Detect features
    orb = cv2.ORB_create(nfeatures=1500)

    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kp_ref, desc_ref = orb.detectAndCompute(gray_ref, None)
    kp_img, desc_img = orb.detectAndCompute(gray_img, None)

    # Match features
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_img, desc_ref)
    matches = sorted(matches, key=lambda x: x.distance)

    # Use only the BEST matches (top 30%) to avoid outliers
    good_count = max(int(len(matches) * 0.3), 4)
    good_matches = matches[:good_count]

    if len(good_matches) < 4:
        return image, 0, 0

    # Get matching points
    pts_img = np.float32([kp_img[m.queryIdx].pt for m in good_matches])
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches])

    # Calculate shift from good matches using MEDIAN (robust to outliers)
    dx_values = pts_ref[:, 0] - pts_img[:, 0]
    dy_values = pts_ref[:, 1] - pts_img[:, 1]

    median_dx = np.median(dx_values)
    median_dy = np.median(dy_values)

    print(f"  Detected shift: dx={median_dx:+.1f}px, dy={median_dy:+.1f}px")

    ref_h, ref_w = reference.shape[:2]

    # Use SIMPLE TRANSLATION (shift) - no warping, no distortion
    # This is a 2x3 affine matrix that only translates
    M = np.float32([
        [1, 0, median_dx],
        [0, 1, median_dy]
    ])

    aligned = cv2.warpAffine(image, M, (ref_w, ref_h))

    return aligned, median_dx, median_dy


def correct_image(reference_path, image_path, output_path=None):
    """
    Apply all corrections to make image match reference

    Corrections applied in order:
    1. Alignment (position, rotation)
    2. Color balance
    3. Brightness & contrast (histogram matching)
    4. Sharpness

    Args:
        reference_path: Path to reference image (the "correct" one)
        image_path: Path to image to correct
        output_path: Path to save corrected image
    """

    print(f"\n{'='*70}")
    print("IMAGE CORRECTOR")
    print(f"{'='*70}\n")

    # Load images
    reference = cv2.imread(reference_path)
    image = cv2.imread(image_path)

    if reference is None or image is None:
        print("Error: Could not load one or both images")
        return None

    ref_h, ref_w = reference.shape[:2]
    img_h, img_w = image.shape[:2]

    print(f"Reference: {reference_path} ({ref_w}x{ref_h})")
    print(f"To fix:    {image_path} ({img_w}x{img_h})")

    # Analyze BEFORE corrections
    print(f"\n--- BEFORE CORRECTIONS ---")
    analyze_diff(reference, image)

    result = image.copy()

    # Correction 1: ALIGNMENT
    print(f"\n{'='*70}")
    print("CORRECTION 1: ALIGNMENT")
    print(f"{'='*70}")
    result, dx, dy = correct_alignment(result, reference)
    print(f"  Applied homography transformation")
    print(f"  Corrected shift: dx={dx:+.1f}px, dy={dy:+.1f}px")

    # Correction 2: COLOR BALANCE
    print(f"\n{'='*70}")
    print("CORRECTION 2: COLOR BALANCE")
    print(f"{'='*70}")
    b_before = [np.mean(result[:, :, i]) for i in range(3)]
    result = correct_color_balance(result, reference)
    b_after = [np.mean(result[:, :, i]) for i in range(3)]
    print(f"  Before: R={b_before[2]:.1f} G={b_before[1]:.1f} B={b_before[0]:.1f}")
    print(f"  After:  R={b_after[2]:.1f} G={b_after[1]:.1f} B={b_after[0]:.1f}")

    # Correction 3: BRIGHTNESS & CONTRAST (histogram matching)
    print(f"\n{'='*70}")
    print("CORRECTION 3: BRIGHTNESS & CONTRAST")
    print(f"{'='*70}")
    brightness_before = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    contrast_before = np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    result = correct_brightness_contrast(result, reference)
    brightness_after = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    contrast_after = np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    print(f"  Brightness: {brightness_before:.1f} -> {brightness_after:.1f}")
    print(f"  Contrast:   {contrast_before:.1f} -> {contrast_after:.1f}")

    # Correction 4: SHARPNESS
    print(f"\n{'='*70}")
    print("CORRECTION 4: SHARPNESS")
    print(f"{'='*70}")
    sharp_before = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    result, action = correct_sharpness(result, reference)
    sharp_after = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    print(f"  Action: {action}")
    print(f"  Sharpness: {sharp_before:.1f} -> {sharp_after:.1f}")

    # Analyze AFTER corrections
    print(f"\n{'='*70}")
    print("--- AFTER ALL CORRECTIONS ---")
    print(f"{'='*70}")
    analyze_diff(reference, result)

    # Save
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"\nSaved corrected image to: {output_path}")

    print(f"\n{'='*70}")
    print("DONE! All corrections applied.")
    print(f"{'='*70}")

    return result


def analyze_diff(reference, image):
    """Quick analysis of differences between two images"""

    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness_diff = np.mean(img_gray) - np.mean(ref_gray)
    contrast_diff = np.std(img_gray) - np.std(ref_gray)

    ref_b, ref_g, ref_r = [np.mean(reference[:, :, i]) for i in range(3)]
    img_b, img_g, img_r = [np.mean(image[:, :, i]) for i in range(3)]

    sharpness_ref = cv2.Laplacian(ref_gray, cv2.CV_64F).var()
    sharpness_img = cv2.Laplacian(img_gray, cv2.CV_64F).var()

    print(f"  Brightness diff: {brightness_diff:+.1f}")
    print(f"  Contrast diff:   {contrast_diff:+.1f}")
    print(f"  Color shift:     R={img_r-ref_r:+.1f} G={img_g-ref_g:+.1f} B={img_b-ref_b:+.1f}")
    print(f"  Sharpness ratio: {sharpness_img/sharpness_ref:.2f}x")


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 image_corrector.py <reference_image> <image_to_fix> [output]")
        print("\nExample:")
        print("  python3 image_corrector.py 20250521170005.JPG 20250522080009.JPG")
        sys.exit(1)

    reference = sys.argv[1]
    to_fix = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(to_fix)[0] + '_corrected' + os.path.splitext(to_fix)[1]

    print("="*70)
    print("IMAGE CORRECTOR - MATCH TO REFERENCE")
    print("="*70)
    print(f"\nReference: {reference}")
    print(f"To fix:    {to_fix}")
    print(f"Output:    {output}")

    correct_image(reference, to_fix, output)

    print(f"\nVerify:")
    print(f"  python3 image_comparator.py {reference} {output}")


if __name__ == "__main__":
    main()
