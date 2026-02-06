# Feature Detection Demo

A simple script to understand how OpenCV detects features in images.

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

Supports all common image formats: **JPG, PNG, BMP, TIFF, WebP**, etc.

```bash
# Basic usage (detects 1000 features by default)
python feature_detector.py path/to/your/image.jpg
python feature_detector.py path/to/your/image.png
python feature_detector.py path/to/your/photo.bmp

# Detect more features (more detailed)
python feature_detector.py path/to/your/image.png 2000

# Detect fewer features (faster)
python feature_detector.py path/to/your/image.tiff 500
```

## What It Does

1. Loads your image
2. Converts it to grayscale
3. Detects distinctive points (corners, edges, patterns)
4. Draws green circles on detected features
5. Saves output as `image_features.jpg`
6. Prints statistics about detected features

## Understanding the Output

- **Green circles**: Locations of detected features
- **Lines from circles**: Direction/orientation of the feature
- **Larger circles**: More prominent/stronger features

## Next Steps

Try running it on different images:
- Photos with lots of detail (buildings, nature) → More features
- Photos with few details (sky, plain walls) → Fewer features
- Different formats (PNG, JPG, TIFF) → All work the same
- Compare how many features are detected in each

Photos with more features are easier to align and stabilize!
