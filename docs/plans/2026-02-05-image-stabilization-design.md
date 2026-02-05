# Image Stabilization & Video Creation - Design Document

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
