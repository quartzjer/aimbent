#!/usr/bin/env python3
import argparse
from dotenv import load_dotenv
from PIL import Image
from screen_compare import compare_images
import gemini_look

def main():
    parser = argparse.ArgumentParser(description="Find differences between two images using Gemini API.")
    parser.add_argument("image1", help="Path to the first image file")
    parser.add_argument("image2", help="Path to the second image file")
    args = parser.parse_args()

    try:
        im1 = Image.open(args.image1).convert("RGB")
        im2 = Image.open(args.image2).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error opening image files: {e}")

    # Pixel-based comparison; assume compare_images returns native coordinates.
    compare_boxes = compare_images(im1, im2)
    if not compare_boxes:
        print("No significant differences found.")
        return

    # Determine largest box based on native pixel area.
    largest_box = max(compare_boxes, key=lambda b: (b["box_2d"][3] - b["box_2d"][1]) * (b["box_2d"][2] - b["box_2d"][0]))
    result = gemini_look.gemini_describe_region(im2, largest_box)
    if result:
        print(result)
    else:
        print("Gemini call failed.")

if __name__ == '__main__':
    main()
