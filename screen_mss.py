import io

import mss
from PIL import Image

def capture_screenshot():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # monitor 1 is typically the primary monitor
        screenshot = sct.grab(monitor)
    # Convert MSS screenshot to a PIL Image (note that MSS returns BGRA data)
    img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
    # Save the image to a BytesIO object in PNG format
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    # Return the bytes data
    return buf.getvalue()

def main():
    png_data = capture_screenshot()
    print(f"Captured screenshot of {len(png_data)} bytes")
    # You can now use png_data directly in memory (for example, sending over network or further analysis)

if __name__ == '__main__':
    main()
