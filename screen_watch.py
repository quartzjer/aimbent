#!/usr/bin/env python3
import argparse
import os
import time
import datetime
from PIL import ImageDraw
from screen_dbus import screen_snap
from screen_compare import compare_images

def process_screenshots(interval, output_dir, verbose, min_threshold):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Store previous PIL images for each monitor.
    prev_images = {}

    while True:
        cycle_start = time.time()
        monitor_images = screen_snap()
        for idx, pil_img in enumerate(monitor_images, start=1):
            if idx not in prev_images:
                prev_images[idx] = pil_img
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(output_dir, f"monitor_{idx}_{timestamp}.png")
                pil_img.save(filename)
                if verbose:
                    print(f"[Monitor {idx}] Initial screenshot saved: {filename}")
            else:
                prev_img = prev_images[idx]
                if prev_img.size != pil_img.size:
                    if verbose:
                        print(f"[Monitor {idx}] Size mismatch; updating reference image.")
                    prev_images[idx] = pil_img
                    continue

                boxes = compare_images(prev_img, pil_img)
                if boxes:
                    # Select the largest bounding box.
                    largest_box = max(boxes, key=lambda b: (b["box_2d"][3] - b["box_2d"][1]) * (b["box_2d"][2] - b["box_2d"][0]))
                    y_min, x_min, y_max, x_max = largest_box["box_2d"]
                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    if box_width > min_threshold and box_height > min_threshold:
                        if verbose:
                            print(f"[Monitor {idx}] Detected significant difference: {largest_box}")
                        annotated = pil_img.copy()
                        draw = ImageDraw.Draw(annotated)
                        draw.rectangle(((x_min, y_min), (x_max, y_max)), outline="red", width=3)
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(output_dir, f"monitor_{idx}_{timestamp}_diff.png")
                        annotated.save(filename)
                        print(f"[Monitor {idx}] Saved annotated diff image: {filename}")
                        prev_images[idx] = pil_img  # Update reference image.
                    else:
                        if verbose:
                            print(f"[Monitor {idx}] Difference detected but largest box {largest_box} is smaller than threshold.")
                else:
                    if verbose:
                        print(f"[Monitor {idx}] No significant change detected.")
        elapsed = time.time() - cycle_start
        sleep_time = max(0, interval - elapsed)
        if verbose:
            print(f"Sleeping for {sleep_time:.2f} seconds before next cycle...\n")
        time.sleep(sleep_time)

def main():
    parser = argparse.ArgumentParser(
        description="Periodically capture screenshots and save updated regions using a comparison approach."
    )
    parser.add_argument("interval", type=float, help="Seconds between screenshots")
    parser.add_argument("directory", type=str, help="Directory to save screenshots")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--min", type=int, default=400, help="Minimum size threshold for a bounding box (pixels)")
    args = parser.parse_args()
    process_screenshots(args.interval, args.directory, args.verbose, args.min)

if __name__ == '__main__':
    main()
