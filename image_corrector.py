#!/usr/bin/env python3
"""
Image Corrector - Fix color, contrast, sharpness, and alignment differences
between two images so they can be used for smooth video generation
"""

import cv2
import numpy as np
import sys
import os


def match_histogram_channel(source, reference):
    """
    Match histogram of a single channel from source to reference
    using CDF (cumulative distribution function) mapping
    """

    ref_hist, _ = np.histogram(reference.flatten(), 256, [0, 256])
    src_hist, _ = np.histogram(source.flatten(), 256, [0, 256])

    ref_cdf = ref_hist.cumsum()
    src_cdf = src_hist.cumsum()

    ref_cdf = ref_cdf / ref_cdf[-1]
    src_cdf = src_cdf / src_cdf[-1]

    lookup = np.zeros(256, dtype=np.uint8)
    for val in range(256):
        closest = np.argmin(np.abs(ref_cdf - src_cdf[val]))
        lookup[val] = closest

    return lookup[source]


def detect_haze_level(image):
    """
    Measure haze level using Dark Channel Prior.

    Returns:
        dc_mean: dark channel mean intensity (0-255, higher = more haze)
        is_hazy: True if dehazing is needed
        level: "clear", "slight", "moderate", "heavy"
    """

    min_channel = np.min(image, axis=2)
    patch_size = 15
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)
    dc_mean = float(np.mean(dark_channel))

    if dc_mean < 30:
        return dc_mean, False, "clear"
    elif dc_mean < 60:
        return dc_mean, True, "slight"
    elif dc_mean < 100:
        return dc_mean, True, "moderate"
    else:
        return dc_mean, True, "heavy"


def dehaze_dark_channel_prior(image, omega=0.95, t_min=0.1, patch_size=15):
    """
    Remove haze/cloudiness using the Dark Channel Prior algorithm.

    The physics of haze (atmospheric scattering model):
        I(x) = J(x) * t(x) + A * (1 - t(x))

    Where:
        I(x) = what the camera sees (hazy image)
        J(x) = the clear scene (what we want to recover)
        t(x) = transmission map (how much light gets through at each pixel)
               - t=1 means no haze (nearby objects)
               - t=0 means full haze (very distant objects)
        A    = atmospheric light (color of the haze, usually bright white)

    Rearranging to recover the clear image:
        J(x) = (I(x) - A) / max(t(x), t_min) + A

    Args:
        image: Hazy BGR image
        omega: How much haze to remove (0.0-1.0). 0.95 removes 95% of haze.
               Keep slightly below 1.0 to preserve a natural look.
        t_min: Minimum transmission value to avoid division artifacts
               in heavily hazed areas. Lower = more aggressive dehazing.
        patch_size: Size of the local patch for dark channel computation.

    Returns:
        Dehazed image
    """

    img = image.astype(np.float64) / 255.0
    h, w, c = img.shape

    # ---- Step 1: Compute Dark Channel ----
    # For each pixel, take minimum across R,G,B, then minimum in local patch
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark_channel = cv2.erode(min_channel, kernel)

    # ---- Step 2: Estimate Atmospheric Light (A) ----
    # Pick the brightest pixels in the dark channel (top 0.1%)
    # These are the most haze-affected pixels
    num_pixels = h * w
    num_brightest = max(int(num_pixels * 0.001), 1)

    # Flatten dark channel and find indices of brightest pixels
    dc_flat = dark_channel.ravel()
    indices = np.argsort(dc_flat)[-num_brightest:]

    # Among those pixels, pick the one with highest intensity in original image
    img_flat = img.reshape(num_pixels, 3)
    atmospheric_light = np.zeros(3)
    max_intensity = 0
    for idx in indices:
        pixel_intensity = np.sum(img_flat[idx])
        if pixel_intensity > max_intensity:
            max_intensity = pixel_intensity
            atmospheric_light = img_flat[idx]

    # ---- Step 3: Estimate Transmission Map t(x) ----
    # Normalize image by atmospheric light
    # t(x) = 1 - omega * dark_channel(I(x) / A)
    normalized = np.zeros_like(img)
    for i in range(3):
        normalized[:, :, i] = img[:, :, i] / max(atmospheric_light[i], 0.01)

    normalized_min = np.min(normalized, axis=2)
    transmission = 1.0 - omega * cv2.erode(normalized_min, kernel)

    # ---- Step 4: Refine Transmission Map ----
    # Use guided filter for edge-preserving smoothing
    # This keeps edges sharp while smoothing the transmission map
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
    transmission_refined = _guided_filter(gray, transmission, radius=60, epsilon=0.001)

    # Clamp transmission to minimum value
    transmission_refined = np.clip(transmission_refined, t_min, 1.0)

    # ---- Step 5: Recover Clear Image ----
    # J(x) = (I(x) - A) / max(t(x), t_min) + A
    result = np.zeros_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atmospheric_light[i]) / transmission_refined + atmospheric_light[i]

    # Clip to valid range and convert back to uint8
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return result


def _guided_filter(guide, src, radius, epsilon):
    """
    Edge-preserving guided filter for refining the transmission map.

    Without this, the transmission map is blocky (due to the patch-based
    dark channel). The guided filter smooths it while preserving edges
    from the original image, so dehazing doesn't blur object boundaries.

    Args:
        guide: Guidance image (grayscale original image)
        src: Input to filter (raw transmission map)
        radius: Filter window radius
        epsilon: Regularization (smaller = more faithful to guide edges)

    Returns:
        Filtered transmission map
    """

    mean_guide = cv2.boxFilter(guide, -1, (radius, radius))
    mean_src = cv2.boxFilter(src, -1, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, -1, (radius, radius))
    mean_guide_sq = cv2.boxFilter(guide * guide, -1, (radius, radius))

    cov = mean_guide_src - mean_guide * mean_src
    var = mean_guide_sq - mean_guide * mean_guide

    a = cov / (var + epsilon)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, -1, (radius, radius))
    mean_b = cv2.boxFilter(b, -1, (radius, radius))

    return mean_a * guide + mean_b


def correct_brightness_contrast(image, reference):
    """
    Match brightness and contrast using LAB color space

    LAB separates luminance (L) from color (A, B):
    - L channel: Controls brightness and contrast
    - A channel: Green-Red axis
    - B channel: Blue-Yellow axis

    By matching only the L channel histogram, we fix brightness
    and contrast WITHOUT affecting colors.
    """

    # Convert to LAB
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    # Match L channel histogram (brightness + contrast)
    img_lab[:, :, 0] = match_histogram_channel(img_lab[:, :, 0], ref_lab[:, :, 0])

    # Convert back to BGR
    result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return result


def correct_color_balance(image, reference):
    """
    Match color balance using LAB color space with FULL histogram matching

    LAB A and B channels control color:
    - A channel: negative = green, positive = red
    - B channel: negative = blue, positive = yellow

    Uses CDF histogram matching (not just mean/std) so that
    EVERY color value is precisely remapped - deep blue stays deep,
    light blue stays light, matching the reference exactly.
    """

    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)

    # Full histogram match on A and B channels
    img_lab[:, :, 1] = match_histogram_channel(img_lab[:, :, 1], ref_lab[:, :, 1])
    img_lab[:, :, 2] = match_histogram_channel(img_lab[:, :, 2], ref_lab[:, :, 2])

    result = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)
    return result


def correct_saturation(image, reference, strength=0.85):
    """
    Match saturation using HSV color space with BLENDED histogram matching

    HSV separates:
    - H (Hue): The actual color (red, green, blue)
    - S (Saturation): How vivid/muted the color is
    - V (Value): Brightness

    Uses CDF histogram matching on S channel, then blends with the
    original saturation to prevent over-boosting vibrant areas
    (e.g., green vegetation becoming unnaturally vivid).

    Args:
        strength: 0.0 = keep original saturation, 1.0 = full histogram match
                  Default 0.85 = 85% matched + 15% original (prevents over-saturation)
    """

    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)

    # Save original saturation
    original_sat = img_hsv[:, :, 1].copy()

    # Full histogram match on Saturation channel
    matched_sat = match_histogram_channel(img_hsv[:, :, 1], ref_hsv[:, :, 1])

    # Blend: strength% matched + (1-strength)% original
    img_hsv[:, :, 1] = np.clip(
        strength * matched_sat.astype(np.float32) +
        (1 - strength) * original_sat.astype(np.float32),
        0, 255
    ).astype(np.uint8)

    result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return result


def correct_final_bgr_pass(image, reference, strength=0.85):
    """
    Final pass: Blended BGR histogram matching as catch-all

    After LAB and HSV corrections, there may still be small
    per-channel differences. This final BGR pass nudges
    each channel toward the reference distribution.

    Uses blending to prevent compounding over-correction
    from previous steps (which can over-saturate colors).

    Args:
        strength: 0.0 = keep current, 1.0 = full histogram match
                  Default 0.85 = close match without over-correcting
    """

    result = image.copy()

    for i in range(3):  # B, G, R channels
        matched = match_histogram_channel(image[:, :, i], reference[:, :, i])
        result[:, :, i] = np.clip(
            strength * matched.astype(np.float32) +
            (1 - strength) * image[:, :, i].astype(np.float32),
            0, 255
        ).astype(np.uint8)

    return result


def detect_sky_mask(image):
    """
    Detect sky region in the image using brightness and saturation.

    Sky pixels are typically:
    - High brightness (L > 170 in LAB)
    - Low saturation (S < 60 in HSV)
    - Located in the upper portion of the image

    Returns:
        sky_mask: float32 mask where 1.0 = sky, 0.0 = non-sky
                  with smooth gradient at the boundary
    """

    height, width = image.shape[:2]

    # Convert to LAB and HSV
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    l_channel = lab[:, :, 0]  # Brightness
    s_channel = hsv[:, :, 1]  # Saturation

    # Sky = bright + low saturation
    bright_mask = l_channel > 170
    low_sat_mask = s_channel < 60

    # Combine: must be both bright AND low saturation
    sky_mask = (bright_mask & low_sat_mask).astype(np.float32)

    # Only consider upper 60% of image as potential sky
    cutoff_row = int(height * 0.6)
    sky_mask[cutoff_row:, :] = 0

    # Smooth the mask edges with Gaussian blur for gradual transition
    sky_mask = cv2.GaussianBlur(sky_mask, (51, 51), 0)

    sky_pixels = np.sum(sky_mask > 0.5)
    total_pixels = height * width
    sky_pct = (sky_pixels / total_pixels) * 100

    print(f"  Sky region detected: {sky_pct:.1f}% of image")

    return sky_mask


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
    1. Dehazing (Dark Channel Prior - only if haze detected)
    2. Alignment (affine translation)
    3. Brightness & contrast (LAB L-channel histogram matching)
    4. Color balance & temperature (LAB A+B channel histogram matching)
    5. Saturation (HSV S-channel histogram matching)
    6. Final BGR pass (catch-all per-channel histogram matching)
    7. Sharpness (unsharp mask or Gaussian blur)
    8. Sky-aware blending (neutralize yellow tint on clouds/sky)

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

    # Correction 1: DEHAZING (only if haze detected)
    print(f"\n{'='*70}")
    print("CORRECTION 1: DEHAZING (Dark Channel Prior)")
    print(f"{'='*70}")
    img_dc, img_hazy, img_haze_level = detect_haze_level(result)
    ref_dc, ref_hazy, ref_haze_level = detect_haze_level(reference)
    print(f"  Image haze:     {img_haze_level.upper()} (dark channel: {img_dc:.1f})")
    print(f"  Reference haze: {ref_haze_level.upper()} (dark channel: {ref_dc:.1f})")

    if img_hazy and img_dc > ref_dc + 10:
        # Image is hazier than reference - dehaze it
        # Adjust omega based on how much hazier the image is vs reference
        haze_diff = img_dc - ref_dc
        if haze_diff > 70:
            omega = 0.95  # Heavy difference - aggressive dehazing
        elif haze_diff > 40:
            omega = 0.85  # Moderate difference
        else:
            omega = 0.70  # Slight difference - gentle dehazing

        print(f"  Image is HAZIER than reference (diff: {haze_diff:.1f})")
        print(f"  Applying dehazing with omega={omega:.2f}...")
        result = dehaze_dark_channel_prior(result, omega=omega)
        new_dc, _, new_level = detect_haze_level(result)
        print(f"  After dehazing: {new_level.upper()} (dark channel: {new_dc:.1f})")
    else:
        print(f"  No dehazing needed (image is clear or clearer than reference)")

    # Correction 2: ALIGNMENT
    print(f"\n{'='*70}")
    print("CORRECTION 2: ALIGNMENT")
    print(f"{'='*70}")
    result, dx, dy = correct_alignment(result, reference)
    print(f"  Applied affine translation")
    print(f"  Corrected shift: dx={dx:+.1f}px, dy={dy:+.1f}px")

    # Correction 3: BRIGHTNESS & CONTRAST (LAB L-channel histogram matching)
    print(f"\n{'='*70}")
    print("CORRECTION 3: BRIGHTNESS & CONTRAST (LAB color space)")
    print(f"{'='*70}")
    brightness_before = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    contrast_before = np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    result = correct_brightness_contrast(result, reference)
    brightness_after = np.mean(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    contrast_after = np.std(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY))
    print(f"  Brightness: {brightness_before:.1f} -> {brightness_after:.1f}")
    print(f"  Contrast:   {contrast_before:.1f} -> {contrast_after:.1f}")

    # Correction 4: COLOR BALANCE / TEMPERATURE (LAB A+B channels)
    print(f"\n{'='*70}")
    print("CORRECTION 4: COLOR BALANCE & TEMPERATURE (LAB color space)")
    print(f"{'='*70}")
    img_lab_before = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float64)
    a_before = np.mean(img_lab_before[:, :, 1])
    b_before = np.mean(img_lab_before[:, :, 2])
    a_ref = np.mean(ref_lab[:, :, 1])
    b_ref = np.mean(ref_lab[:, :, 2])
    print(f"  Before: A(green-red)={a_before:.1f} B(blue-yellow)={b_before:.1f}")
    print(f"  Target: A(green-red)={a_ref:.1f} B(blue-yellow)={b_ref:.1f}")
    result = correct_color_balance(result, reference)
    img_lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float64)
    a_after = np.mean(img_lab_after[:, :, 1])
    b_after = np.mean(img_lab_after[:, :, 2])
    print(f"  After:  A(green-red)={a_after:.1f} B(blue-yellow)={b_after:.1f}")

    # Correction 5: SATURATION (HSV S-channel)
    print(f"\n{'='*70}")
    print("CORRECTION 5: SATURATION (HSV color space)")
    print(f"{'='*70}")
    img_hsv_before = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    sat_before = np.mean(img_hsv_before[:, :, 1])
    sat_ref = np.mean(ref_hsv[:, :, 1])
    result = correct_saturation(result, reference)
    img_hsv_after = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
    sat_after = np.mean(img_hsv_after[:, :, 1])
    print(f"  Before:    {sat_before:.1f}")
    print(f"  Reference: {sat_ref:.1f}")
    print(f"  After:     {sat_after:.1f}")
    print(f"  Blend:     85% matched + 15% original (close match, no over-saturation)")

    # Correction 6: FINAL BGR PASS (catch-all)
    print(f"\n{'='*70}")
    print("CORRECTION 6: FINAL BGR PASS (catch-all)")
    print(f"{'='*70}")
    ref_b, ref_g, ref_r = [np.mean(reference[:, :, i]) for i in range(3)]
    img_b, img_g, img_r = [np.mean(result[:, :, i]) for i in range(3)]
    print(f"  Before: R={img_r:.1f} G={img_g:.1f} B={img_b:.1f}")
    print(f"  Target: R={ref_r:.1f} G={ref_g:.1f} B={ref_b:.1f}")
    result = correct_final_bgr_pass(result, reference)
    img_b2, img_g2, img_r2 = [np.mean(result[:, :, i]) for i in range(3)]
    print(f"  After:  R={img_r2:.1f} G={img_g2:.1f} B={img_b2:.1f}")
    print(f"  Blend:  85% matched + 15% current (close match, no over-correction)")

    # Correction 7: SHARPNESS
    print(f"\n{'='*70}")
    print("CORRECTION 7: SHARPNESS")
    print(f"{'='*70}")
    sharp_before = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    result, action = correct_sharpness(result, reference)
    sharp_after = cv2.Laplacian(cv2.cvtColor(result, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    print(f"  Action: {action}")
    print(f"  Sharpness: {sharp_before:.1f} -> {sharp_after:.1f}")

    # Correction 7: SKY-AWARE BLENDING (remove yellow tint from clouds)
    print(f"\n{'='*70}")
    print("CORRECTION 7: SKY-AWARE BLENDING (neutralize sky tint)")
    print(f"{'='*70}")

    # Detect sky in the ORIGINAL image (before corrections shifted colors)
    sky_mask = detect_sky_mask(image)

    # For sky pixels: blend back toward a neutral version
    # The fully corrected result has yellow tint in sky from global histogram matching
    # Fix: in sky regions, only keep brightness correction but use original color tone
    sky_neutral = correct_brightness_contrast(image, reference)  # brightness-only correction
    sky_neutral, _ = correct_sharpness(sky_neutral, reference)   # match sharpness too

    # Blend: sky regions use neutral version, ground uses full correction
    sky_mask_3ch = np.stack([sky_mask] * 3, axis=-1)
    # Sky gets 70% neutral + 30% full correction (keeps some matching but removes tint)
    sky_blend = cv2.addWeighted(sky_neutral, 0.7, result, 0.3, 0)
    result = (sky_mask_3ch * sky_blend + (1 - sky_mask_3ch) * result).astype(np.uint8)

    print(f"  Applied neutral color tone to sky region")
    print(f"  Sky blend: 70% brightness-only + 30% full correction")

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


def match_histogram_to_average(channel1, channel2):
    """
    Compute the average CDF of two channels and map both to it.

    Instead of matching one image to the other, both images are
    corrected toward a shared midpoint — their average histogram.

    Returns:
        (corrected_ch1, corrected_ch2)
    """

    hist1, _ = np.histogram(channel1.flatten(), 256, [0, 256])
    hist2, _ = np.histogram(channel2.flatten(), 256, [0, 256])

    # Average histogram = midpoint of both distributions
    avg_hist = (hist1.astype(np.float64) + hist2.astype(np.float64)) / 2.0

    cdf1 = hist1.cumsum().astype(np.float64)
    cdf2 = hist2.cumsum().astype(np.float64)
    avg_cdf = avg_hist.cumsum()

    cdf1 = cdf1 / cdf1[-1]
    cdf2 = cdf2 / cdf2[-1]
    avg_cdf = avg_cdf / avg_cdf[-1]

    # Map channel1 -> average
    lookup1 = np.zeros(256, dtype=np.uint8)
    for val in range(256):
        lookup1[val] = np.argmin(np.abs(avg_cdf - cdf1[val]))

    # Map channel2 -> average
    lookup2 = np.zeros(256, dtype=np.uint8)
    for val in range(256):
        lookup2[val] = np.argmin(np.abs(avg_cdf - cdf2[val]))

    return lookup1[channel1], lookup2[channel2]


def correct_image_pair(image1_path, image2_path, output1_path=None, output2_path=None):
    """
    Mutual correction: correct BOTH images toward their average.

    Instead of forcing image2 to match image1 (one-sided), this finds
    the midpoint of both images and corrects each toward it. Result:
    both images look virtually identical.

    Corrections applied to both:
    1. Alignment (image2 aligned to image1, image1 unchanged)
    2. Brightness & contrast (LAB L-channel -> average)
    3. Color balance (LAB A+B channels -> average)
    4. Saturation (HSV S-channel -> average)
    5. Final BGR pass (per-channel -> average)
    6. Sharpness (match to average sharpness)
    7. Sky-aware blending (neutralize tint on clouds)
    """

    print(f"\n{'='*70}")
    print("MUTUAL IMAGE CORRECTOR (meet in the middle)")
    print(f"{'='*70}\n")

    # Load images
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    if img1 is None or img2 is None:
        print("Error: Could not load one or both images")
        return None, None

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    print(f"Image 1: {image1_path} ({w1}x{h1})")
    print(f"Image 2: {image2_path} ({w2}x{h2})")

    # Analyze BEFORE
    print(f"\n--- BEFORE CORRECTIONS ---")
    analyze_diff(img1, img2)

    result1 = img1.copy()
    result2 = img2.copy()

    # Step 1: ALIGNMENT (align image2 to image1 — only one needs shifting)
    print(f"\n{'='*70}")
    print("STEP 1: ALIGNMENT (align Image 2 to Image 1)")
    print(f"{'='*70}")
    result2, dx, dy = correct_alignment(result2, result1)
    print(f"  Corrected shift: dx={dx:+.1f}px, dy={dy:+.1f}px")

    # Step 2: BRIGHTNESS & CONTRAST (LAB L-channel -> average)
    print(f"\n{'='*70}")
    print("STEP 2: BRIGHTNESS & CONTRAST (both -> average)")
    print(f"{'='*70}")
    lab1 = cv2.cvtColor(result1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(result2, cv2.COLOR_BGR2LAB)
    b_before1 = np.mean(lab1[:, :, 0])
    b_before2 = np.mean(lab2[:, :, 0])
    lab1[:, :, 0], lab2[:, :, 0] = match_histogram_to_average(lab1[:, :, 0], lab2[:, :, 0])
    b_after1 = np.mean(lab1[:, :, 0])
    b_after2 = np.mean(lab2[:, :, 0])
    result1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    result2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    print(f"  Image 1 brightness: {b_before1:.1f} -> {b_after1:.1f}")
    print(f"  Image 2 brightness: {b_before2:.1f} -> {b_after2:.1f}")

    # Step 3: COLOR BALANCE (LAB A+B channels -> average)
    print(f"\n{'='*70}")
    print("STEP 3: COLOR BALANCE (both -> average)")
    print(f"{'='*70}")
    lab1 = cv2.cvtColor(result1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(result2, cv2.COLOR_BGR2LAB)
    a1_before = np.mean(lab1[:, :, 1])
    a2_before = np.mean(lab2[:, :, 1])
    lab1[:, :, 1], lab2[:, :, 1] = match_histogram_to_average(lab1[:, :, 1], lab2[:, :, 1])
    lab1[:, :, 2], lab2[:, :, 2] = match_histogram_to_average(lab1[:, :, 2], lab2[:, :, 2])
    a1_after = np.mean(lab1[:, :, 1])
    a2_after = np.mean(lab2[:, :, 1])
    result1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)
    result2 = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    print(f"  Image 1 color A: {a1_before:.1f} -> {a1_after:.1f}")
    print(f"  Image 2 color A: {a2_before:.1f} -> {a2_after:.1f}")

    # Step 4: SATURATION (HSV S-channel -> average)
    print(f"\n{'='*70}")
    print("STEP 4: SATURATION (both -> average)")
    print(f"{'='*70}")
    hsv1 = cv2.cvtColor(result1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(result2, cv2.COLOR_BGR2HSV)
    s1_before = np.mean(hsv1[:, :, 1])
    s2_before = np.mean(hsv2[:, :, 1])
    hsv1[:, :, 1], hsv2[:, :, 1] = match_histogram_to_average(hsv1[:, :, 1], hsv2[:, :, 1])
    s1_after = np.mean(hsv1[:, :, 1])
    s2_after = np.mean(hsv2[:, :, 1])
    result1 = cv2.cvtColor(hsv1, cv2.COLOR_HSV2BGR)
    result2 = cv2.cvtColor(hsv2, cv2.COLOR_HSV2BGR)
    print(f"  Image 1 saturation: {s1_before:.1f} -> {s1_after:.1f}")
    print(f"  Image 2 saturation: {s2_before:.1f} -> {s2_after:.1f}")

    # Step 5: FINAL BGR PASS (per-channel -> average)
    print(f"\n{'='*70}")
    print("STEP 5: FINAL BGR PASS (both -> average)")
    print(f"{'='*70}")
    for i, name in enumerate(['B', 'G', 'R']):
        before1 = np.mean(result1[:, :, i])
        before2 = np.mean(result2[:, :, i])
        result1[:, :, i], result2[:, :, i] = match_histogram_to_average(
            result1[:, :, i], result2[:, :, i]
        )
        after1 = np.mean(result1[:, :, i])
        after2 = np.mean(result2[:, :, i])
        print(f"  {name}: Image1 {before1:.1f}->{after1:.1f}  Image2 {before2:.1f}->{after2:.1f}")

    # Step 6: SHARPNESS (match both to average sharpness)
    print(f"\n{'='*70}")
    print("STEP 6: SHARPNESS (both -> average)")
    print(f"{'='*70}")
    gray1 = cv2.cvtColor(result1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
    sharp1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
    sharp2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
    avg_sharp = (sharp1 + sharp2) / 2

    # Adjust image1 sharpness toward average
    ratio1 = avg_sharp / sharp1 if sharp1 > 0 else 1.0
    if ratio1 > 1.2:
        strength = min(ratio1 - 1.0, 2.0)
        blurred = cv2.GaussianBlur(result1, (0, 0), 3)
        result1 = cv2.addWeighted(result1, 1.0 + strength, blurred, -strength, 0)
        print(f"  Image 1: sharpened ({sharp1:.1f} -> target {avg_sharp:.1f})")
    elif ratio1 < 0.8:
        result1 = cv2.GaussianBlur(result1, (3, 3), 0)
        print(f"  Image 1: blurred ({sharp1:.1f} -> target {avg_sharp:.1f})")
    else:
        print(f"  Image 1: no change ({sharp1:.1f}, target {avg_sharp:.1f})")

    # Adjust image2 sharpness toward average
    ratio2 = avg_sharp / sharp2 if sharp2 > 0 else 1.0
    if ratio2 > 1.2:
        strength = min(ratio2 - 1.0, 2.0)
        blurred = cv2.GaussianBlur(result2, (0, 0), 3)
        result2 = cv2.addWeighted(result2, 1.0 + strength, blurred, -strength, 0)
        print(f"  Image 2: sharpened ({sharp2:.1f} -> target {avg_sharp:.1f})")
    elif ratio2 < 0.8:
        result2 = cv2.GaussianBlur(result2, (3, 3), 0)
        print(f"  Image 2: blurred ({sharp2:.1f} -> target {avg_sharp:.1f})")
    else:
        print(f"  Image 2: no change ({sharp2:.1f}, target {avg_sharp:.1f})")

    # Step 7: SKY-AWARE BLENDING (for both images)
    print(f"\n{'='*70}")
    print("STEP 7: SKY-AWARE BLENDING (neutralize sky tint)")
    print(f"{'='*70}")

    for label, original, result_img in [("Image 1", img1, result1), ("Image 2", img2, result2)]:
        print(f"\n  {label}:")
        sky_mask = detect_sky_mask(original)
        sky_neutral = correct_brightness_contrast(original, result_img)
        sky_mask_3ch = np.stack([sky_mask] * 3, axis=-1)
        sky_blend = cv2.addWeighted(sky_neutral, 0.7, result_img, 0.3, 0)
        blended = (sky_mask_3ch * sky_blend + (1 - sky_mask_3ch) * result_img).astype(np.uint8)
        if label == "Image 1":
            result1 = blended
        else:
            result2 = blended

    # Analyze AFTER
    print(f"\n{'='*70}")
    print("--- AFTER MUTUAL CORRECTIONS ---")
    print(f"{'='*70}")
    analyze_diff(result1, result2)

    # Save
    if output1_path:
        cv2.imwrite(output1_path, result1)
        print(f"\nSaved Image 1 corrected to: {output1_path}")
    if output2_path:
        cv2.imwrite(output2_path, result2)
        print(f"\nSaved Image 2 corrected to: {output2_path}")

    print(f"\n{'='*70}")
    print("DONE! Both images corrected toward their average.")
    print(f"{'='*70}")

    return result1, result2


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

    # LAB color temperature
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB).astype(np.float64)
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    temp_a_diff = np.mean(img_lab[:, :, 1]) - np.mean(ref_lab[:, :, 1])
    temp_b_diff = np.mean(img_lab[:, :, 2]) - np.mean(ref_lab[:, :, 2])

    # HSV saturation
    ref_hsv = cv2.cvtColor(reference, cv2.COLOR_BGR2HSV)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sat_diff = np.mean(img_hsv[:, :, 1].astype(float)) - np.mean(ref_hsv[:, :, 1].astype(float))

    # Haze
    img_dc, img_hazy, img_level = detect_haze_level(image)
    ref_dc, ref_hazy, ref_level = detect_haze_level(reference)

    print(f"  Brightness diff:  {brightness_diff:+.1f}")
    print(f"  Contrast diff:    {contrast_diff:+.1f}")
    print(f"  Color shift:      R={img_r-ref_r:+.1f} G={img_g-ref_g:+.1f} B={img_b-ref_b:+.1f}")
    print(f"  Color temp (LAB): A={temp_a_diff:+.1f} B={temp_b_diff:+.1f}")
    print(f"  Saturation diff:  {sat_diff:+.1f}")
    print(f"  Haze:             image={img_level}({img_dc:.0f}) ref={ref_level}({ref_dc:.0f}) diff={img_dc-ref_dc:+.0f}")
    print(f"  Sharpness ratio:  {sharpness_img/sharpness_ref:.2f}x")


def main():
    # Check for --mutual flag
    args = [a for a in sys.argv[1:] if a != '--mutual']
    mutual_mode = '--mutual' in sys.argv

    if len(args) < 2:
        print("Usage:")
        print("  python3 image_corrector.py <reference> <image_to_fix> [output]")
        print("  python3 image_corrector.py --mutual <image1> <image2>")
        print("\nModes:")
        print("  Default:  Correct image2 to match image1 (one-sided)")
        print("  --mutual: Correct BOTH images toward their average (meet in the middle)")
        print("\nExamples:")
        print("  python3 image_corrector.py 20221101130004.JPG 20221101140005.JPG")
        print("  python3 image_corrector.py --mutual 20221101130004.JPG 20221101140005.JPG")
        sys.exit(1)

    if mutual_mode:
        image1 = args[0]
        image2 = args[1]
        ext1 = os.path.splitext(image1)
        ext2 = os.path.splitext(image2)
        output1 = ext1[0] + '_corrected' + ext1[1]
        output2 = ext2[0] + '_corrected' + ext2[1]

        print("="*70)
        print("IMAGE CORRECTOR - MUTUAL MODE (meet in the middle)")
        print("="*70)
        print(f"\nImage 1:  {image1}")
        print(f"Image 2:  {image2}")
        print(f"Output 1: {output1}")
        print(f"Output 2: {output2}")

        correct_image_pair(image1, image2, output1, output2)

        print(f"\nVerify:")
        print(f"  python3 image_comparator.py {output1} {output2}")

    else:
        reference = args[0]
        to_fix = args[1]
        output = args[2] if len(args) > 2 else os.path.splitext(to_fix)[0] + '_corrected' + os.path.splitext(to_fix)[1]

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
