#!/usr/bin/env python3
"""
Simple Feature Detection Script
Demonstrates how OpenCV detects distinctive points in images
"""

import cv2
import sys
import os


def detect_features(image_path, num_features=1000):
    """
    Detect and visualize features in an image using ORB detector

    Args:
        image_path: Path to the input image
        num_features: Number of features to detect (default: 1000)
    """

    # Step 1: Load the image
    # cv2.imread() supports: JPG, PNG, BMP, TIFF, WebP, and more
    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)


    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")

    # Step 2: Convert to grayscale (features are detected on intensity, not color)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("Converted to grayscale")

    # Step 3: Create ORB detector
    # ORB = Oriented FAST and Rotated BRIEF
    # It finds corners and edges that are distinctive
    orb = cv2.ORB_create(nfeatures=num_features)
    print(f"Created ORB detector (looking for {num_features} features)")

    # Step 4: Detect keypoints and compute descriptors
    # - Keypoints: locations of interesting points (x, y coordinates)
    # - Descriptors: unique "fingerprints" of each keypoint (for matching)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    print(f"Found {len(keypoints)} keypoints")

    # Step 5: Draw keypoints on the image
    # Green circles = detected features
    # Lines coming from circles = orientation (direction the feature is "pointing")
    output_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        color=(0, 255, 0),  # Green color
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS  # Show orientation
    )

    # Step 6: Print some keypoint details
    print("\n=== Sample Keypoints ===")
    for i, kp in enumerate(keypoints[:5]):  # Show first 5
        print(f"Keypoint {i+1}:")
        print(f"  Position: ({kp.pt[0]:.1f}, {kp.pt[1]:.1f})")
        print(f"  Size: {kp.size:.1f} (how large the feature is)")
        print(f"  Angle: {kp.angle:.1f}° (orientation)")
        print(f"  Response: {kp.response:.3f} (how strong/distinctive)")

    # Step 7: Save the output
    output_path = os.path.splitext(image_path)[0] + "_features.jpg"
    cv2.imwrite(output_path, output_image)
    print(f"\nSaved output to: {output_path}")

    # Step 8: Display statistics
    print("\n=== Statistics ===")
    print(f"Total features detected: {len(keypoints)}")
    print(f"Descriptor shape: {descriptors.shape if descriptors is not None else 'None'}")
    print(f"Each descriptor is a {descriptors.shape[1]}-dimensional vector")

    return keypoints, descriptors, output_image


def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python feature_detector.py <image_path> [num_features]")
        print("\nSupports: JPG, PNG, BMP, TIFF, WebP, and other common formats")
        print("\nExamples:")
        print("  python feature_detector.py photo.jpg")
        print("  python feature_detector.py image.png 500")
        print("  python feature_detector.py picture.bmp 2000")
        sys.exit(1)

    image_path = sys.argv[1]
    num_features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    print("=" * 60)
    print("FEATURE DETECTION DEMO")
    print("=" * 60)
    print()

    detect_features(image_path, num_features)

    print("\n" + "=" * 60)
    print("WHAT DO THESE FEATURES MEAN?")
    print("=" * 60)
    print("""
Features (keypoints) are distinctive points in your image like:
- Corners of objects
- Edges and boundaries
- High-contrast spots
- Texture patterns

These points are used to:
1. Match images together (find same objects)
2. Detect if image is rotated/tilted
3. Calculate how to align images
4. Track objects between frames

The more features detected, the more accurately we can:
- Align images
- Detect rotation
- Calculate transformations
    """)


if __name__ == "__main__":
    main()
