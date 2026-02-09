((venv) ) sudibyajyotijena@Sudibyas-Mac-mini Video_generation_pipeline % python3 image_comparator.py 20250521170005.JPG 20250522080009.JPG

======================================================================
COMPARING IMAGES
======================================================================

Loading images...

Image 1 (20250521170005.JPG): 6000x4000 pixels
Image 2 (20250522080009.JPG): 6000x4000 pixels

======================================================================
CROP DETECTION:
======================================================================

   --- Crop Detection for Image 1 (20250521170005.JPG) ---
   ✓ Image is NOT cropped (full content)

   --- Crop Detection for Image 2 (20250522080009.JPG) ---
   ✓ Image is NOT cropped (full content)

======================================================================
IMAGE QUALITY COMPARISON:
======================================================================

   --- BRIGHTNESS (0=black, 255=white) ---
   Image 1: 133.1/255
   Image 2: 129.3/255
   Difference: -3.8
   ✓ Brightness is CONSISTENT

   --- CONTRAST (higher = more contrast) ---
   Image 1: 41.9
   Image 2: 45.5
   Difference: +3.6
   ✓ Contrast is CONSISTENT

   --- COLOR BALANCE (R/G/B channel averages) ---
   Image 1: R=132.7  G=133.5  B=132.6
   Image 2: R=127.2  G=130.4  B=129.3
   Shift:    R=-5.5      G=-3.0      B=-3.2
   ⚠ SLIGHT color shift detected

   --- SHARPNESS (higher = sharper) ---
   Image 1: 41.7
   Image 2: 71.7
   Ratio: 1.72x
   ✗ SIGNIFICANT sharpness difference! (Image 1 is much blurrier)
     → Will cause BLUR FLICKER in video.

   --- EXPOSURE ---
   Image 1: Overexposed=0.2%  Underexposed=0.0%
   Image 2: Overexposed=0.6%  Underexposed=0.0%
   ✓ Exposure is CONSISTENT

Detecting features...
Image 1: 1000 features detected
Image 2: 1000 features detected

Matching features between images...
Found 388 matching features

======================================================================
DIFFERENCES FOUND:
======================================================================

1. SIZE DIFFERENCE:
   Width:  6000 vs 6000 (difference: +0px)
   Height: 4000 vs 4000 (difference: +0px)

2. FEATURE COUNT:
   Image 1: 1000 features
   Image 2: 1000 features
   Difference: +0 features

3. FEATURE MATCHING:
   Matching features: 388
   Match coverage: 38.8% matched

4. MATCH QUALITY:
   Best match quality: 10.0/256 (lower is better)
   Worst match quality: 81.0/256
   Average quality: 37.2/256

5. SAMPLE MATCHING KEYPOINTS (first 5):

   Match 1:
     Image 1 position: (4646.6, 3502.7)
     Image 2 position: (4639.7, 3490.6)
     Position shift: dx=+6.9px, dy=+12.1px
     Match quality: 10.0/256

   Match 2:
     Image 1 position: (2007.0, 2976.0)
     Image 2 position: (2000.0, 2963.0)
     Position shift: dx=+7.0px, dy=+13.0px
     Match quality: 11.0/256

   Match 3:
     Image 1 position: (4553.0, 3452.0)
     Image 2 position: (4545.0, 3439.0)
     Position shift: dx=+8.0px, dy=+13.0px
     Match quality: 12.0/256

   Match 4:
     Image 1 position: (2816.0, 3224.0)
     Image 2 position: (2808.0, 3211.0)
     Position shift: dx=+8.0px, dy=+13.0px
     Match quality: 13.0/256

   Match 5:
     Image 1 position: (3694.5, 2960.1)
     Image 2 position: (3687.6, 2948.0)
     Position shift: dx=+6.9px, dy=+12.1px
     Match quality: 13.0/256

======================================================================
SUMMARY:
======================================================================

Average position shift:
  Horizontal (X): -129.56px
  Vertical (Y):   +1.10px

✗ Images are MISALIGNED (will cause visible shaking in video)

======================================================================
((venv) ) sudibyajyotijena@Sudibyas-Mac-mini Video_generation_pipeline % # Image Stabilization & Video Creation - Design Document

**Date:** 2026-02-05
**Purpose:** Create an algorithm to stabilize/align multiple images and generate a timelapse video

---

## Overview

A Python pipeline using OpenCV that:
1. Takes multiple input images
2. Aligns them to a reference image (corrects rotation + translation)
3. Normalizes lighting across all images
4. Creates a video from the stabilized frames

---

## Architecture

```
image_automation/
├── main.py                 # Entry point - CLI interface
├── stabilizer/
│   ├── __init__.py
│   ├── aligner.py          # Feature detection, matching, transformation
│   ├── lighting.py         # Histogram matching for consistent exposure
│   └── video_writer.py     # Combine frames into video
└── utils/
    ├── __init__.py
    └── io.py               # Image loading, sorting, validation
```

---

## Pipeline Flow

```
Input Images → Sort by name/timestamp
                    ↓
            Load Reference Image
                    ↓
        ┌───────────────────────┐
        │   For each image:     │
        │   1. Align to ref     │
        │   2. Normalize light  │
        └───────────────────────┘
                    ↓
            Write to Video (MP4)
```

---

## Component Details

### 1. Image Aligner (`stabilizer/aligner.py`)

**Class:** `ImageAligner`

**Algorithm:**
1. Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
   - Finds distinctive points (corners, edges) in both images
   - ORB is fast and free (unlike SIFT)
   - Configure with ~5000 features for good coverage

2. Feature Matching
   - Brute-Force matcher with Hamming distance
   - Cross-check enabled for better matches

3. Filter Good Matches
   - Sort matches by distance
   - Keep top matches (minimum 10-15 needed)

4. Compute Homography
   - Calculate 3x3 transformation matrix using RANSAC
   - Matrix encodes rotation + translation + perspective

5. Warp Image
   - Apply transformation using `cv2.warpPerspective()`

**Edge Cases:**
- Too few matches → Skip image or warn user
- Bad homography → Fallback to previous frame's transformation

---

### 2. Lighting Normalizer (`stabilizer/lighting.py`)

**Class:** `LightingNormalizer`

**Algorithm: Histogram Matching in LAB Color Space**

1. Convert image to LAB color space
   - L = Lightness, A = green-red, B = blue-yellow
   - Separates brightness from color

2. Calculate CDFs (Cumulative Distribution Functions)
   - For both reference and input image L channels

3. Map pixel values
   - For each pixel in input, find reference value with same CDF position

4. Convert back to BGR

**Why LAB?**
- Normalizing L channel preserves natural colors
- RGB normalization can cause color shifts

---

### 3. Video Creator (`stabilizer/video_writer.py`)

**Class:** `VideoCreator`

**Settings:**
| Setting | Default | Notes |
|---------|---------|-------|
| Format | MP4 | Most compatible |
| Codec | mp4v | Works everywhere |
| FPS | 30 | Configurable via CLI |

**Process:**
1. Initialize on first frame (get dimensions)
2. Write frames sequentially
3. Release and finalize

**Edge Case:**
- Different dimensions → Resize to match reference

---

### 4. Utilities (`utils/io.py`)

**Functions:**
- `load_images_sorted(directory)` - Load and natural sort images
- `validate_image(path)` - Check if valid image file

---

## CLI Interface

```bash
# Basic usage
python main.py --input ./photos --reference ./photos/001.jpg --output timelapse.mp4

# With custom FPS
python main.py --input ./photos --reference ./photos/001.jpg --output timelapse.mp4 --fps 24
```

**Arguments:**
| Argument | Required | Description |
|----------|----------|-------------|
| `--input` | Yes | Directory containing input images |
| `--reference` | Yes | Path to reference image |
| `--output` | Yes | Output video file path |
| `--fps` | No | Frames per second (default: 30) |

---

## Dependencies

```
opencv-python>=4.5.0
numpy>=1.20.0
```

---

## Future Extensions

- Multiple reference images for different sections
- GPU acceleration for large batches
- Preview mode before final render
- Additional stabilization methods
