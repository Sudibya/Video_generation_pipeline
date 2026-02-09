#!/usr/bin/env python3
"""
Image Aligner - Fill missing/gray areas by copying from reference image
"""

import cv2
import numpy as np
import sys
import os


def detect_gray_region(image):
    """
    Detect where gray/blank area starts by scanning from bottom up
    Uses 3 methods to reliably detect gray fill from corrupted JPEGs

    Returns:
        row number where real content ends
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape

    # Method 1: Row-to-row similarity (gray fill = identical rows)
    consecutive_similar = 0
    for row in range(height - 1, 0, -1):
        row_diff = np.mean(np.abs(
            gray[row, :].astype(float) - gray[row - 1, :].astype(float)
        ))
        if row_diff < 5:
            consecutive_similar += 1
        else:
            break

    if consecutive_similar > 50:
        content_end = height - consecutive_similar
        print(f"  [Row similarity] Found {consecutive_similar} blank rows")
        return content_end

    # Method 2: Color range per row (real photos = wide range, gray = flat)
    for row in range(height - 1, -1, -1):
        pixel_range = int(np.max(gray[row, :])) - int(np.min(gray[row, :]))
        if pixel_range > 80:
            print(f"  [Color range] Content ends at row {row + 1}")
            return row + 1

    # Method 3: RGB channel difference (gray has R=G=B)
    b, g, r = cv2.split(image)
    for row in range(height - 1, -1, -1):
        channel_diff = max(
            abs(float(np.mean(r[row, :])) - float(np.mean(g[row, :]))),
            abs(float(np.mean(r[row, :])) - float(np.mean(b[row, :]))),
            abs(float(np.mean(g[row, :])) - float(np.mean(b[row, :])))
        )
        if channel_diff > 10:
            print(f"  [Channel analysis] Content ends at row {row + 1}")
            return row + 1

    return height  # No gray found


def fill_from_reference(reference_path, cropped_path, output_path=None):
    """
    Fill the gray/missing bottom of cropped image with the bottom
    from the reference image. Simple and direct.

    Args:
        reference_path: Path to full reference image
        cropped_path: Path to image with gray bottom
        output_path: Path to save fixed image
    """

    print(f"\n{'='*70}")
    print("FILLING MISSING AREA FROM REFERENCE IMAGE")
    print(f"{'='*70}\n")

    # Load both images
    ref_image = cv2.imread(reference_path)
    cropped_image = cv2.imread(cropped_path)

    if ref_image is None or cropped_image is None:
        print("Error: Could not load one or both images")
        return None

    ref_h, ref_w = ref_image.shape[:2]
    crop_h, crop_w = cropped_image.shape[:2]

    print(f"Reference (FULL):  {ref_w}x{ref_h} pixels")
    print(f"To fix:            {crop_w}x{crop_h} pixels")

    # Step 1: Detect where the gray area starts
    print(f"\nStep 1: Detecting gray area...")
    content_end = detect_gray_region(cropped_image)
    gray_height = crop_h - content_end

    if gray_height <= 0:
        print("  No gray area detected! Image appears complete.")
        return cropped_image

    print(f"  Real content:  rows 0 to {content_end}")
    print(f"  Gray area:     rows {content_end} to {crop_h} ({gray_height}px)")
    print(f"  Missing:       {(gray_height / crop_h * 100):.1f}% of the image")

    # Step 2: Resize reference if needed to match cropped dimensions
    print(f"\nStep 2: Preparing reference image...")
    if ref_w != crop_w or ref_h != crop_h:
        ref_resized = cv2.resize(ref_image, (crop_w, crop_h))
        print(f"  Resized reference from {ref_w}x{ref_h} to {crop_w}x{crop_h}")
    else:
        ref_resized = ref_image
        print(f"  Same dimensions - no resize needed")

    # Step 3: Build the final image
    print(f"\nStep 3: Building final image...")
    result = cropped_image.copy()

    # Copy the bottom part from reference into the gray area
    result[content_end:crop_h, :] = ref_resized[content_end:crop_h, :]
    print(f"  Copied rows {content_end}-{crop_h} from reference image")

    # Step 4: Smooth blend at the transition (avoid visible seam)
    blend_height = min(80, gray_height, content_end)
    blend_start = content_end - blend_height

    print(f"\nStep 4: Blending transition zone...")
    for row in range(blend_start, content_end):
        # Alpha: 1.0 at top (all original) -> 0.0 at bottom (all reference)
        alpha = (content_end - row) / blend_height
        result[row, :] = cv2.addWeighted(
            cropped_image[row, :], alpha,
            ref_resized[row, :], 1.0 - alpha,
            0
        )
    print(f"  Blended rows {blend_start}-{content_end} ({blend_height}px smooth transition)")

    # Step 5: Save
    print(f"\nStep 5: Saving result...")
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"  Saved to: {output_path}")

    # Summary
    print(f"\n{'='*70}")
    print("DONE!")
    print(f"{'='*70}")
    print(f"  Top part (original):     rows 0 - {blend_start}")
    print(f"  Blend zone:              rows {blend_start} - {content_end}")
    print(f"  Bottom part (from ref):  rows {content_end} - {crop_h}")
    print(f"  Total pixels restored:   {gray_height * crop_w:,} pixels")

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 image_aligner.py <reference_image> <cropped_image> [output]")
        print("\nExample:")
        print("  python3 image_aligner.py 20250608143005.JPG 20250608150010.JPG")
        sys.exit(1)

    reference = sys.argv[1]
    cropped = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(cropped)[0] + '_fixed' + os.path.splitext(cropped)[1]

    print("="*70)
    print("IMAGE ALIGNER - FILL MISSING AREA FROM REFERENCE")
    print("="*70)
    print(f"\nReference (FULL):  {reference}")
    print(f"To fix:            {cropped}")
    print(f"Output:            {output}")

    result = fill_from_reference(reference, cropped, output)

    if result is not None:
        print(f"\nVerify the result:")
        print(f"  python3 image_comparator.py {reference} {output}")


if __name__ == "__main__":
    main()
