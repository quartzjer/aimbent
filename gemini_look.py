import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import ImageDraw

def gemini_describe_region(image, box):
    """
    Crops the image using native pixel coordinates from box,
    computes normalized coordinates once for the Gemini call, and then
    sends both full image and crop to Gemini.
    """
    # Draw bounding box on a copy of the image
    im_with_box = image.copy()
    native_y_min, native_x_min, native_y_max, native_x_max = box["box_2d"]
    draw = ImageDraw.Draw(im_with_box)
    draw.rectangle(((native_x_min, native_y_min), (native_x_max, native_y_max)), outline="red", width=5)

    cropped = im_with_box.crop((native_x_min, native_y_min, native_x_max, native_y_max))
        
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please create a .env file.")
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = (
        "Analyze the provided image and its region cut-out. "
        "Return a JSON object in the format {\"description\": \"...\"} describing the region."
    )
    try:
        response = client.models.generate_content(
            #model="gemini-2.0-pro-exp-02-05",
            model="gemini-2.0-flash",
            contents=[prompt, im_with_box, cropped],
            config=types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=8192,
                response_mime_type="application/json",
                system_instruction="You analyze screenshots, you will be given a full screenshot with a red box around a region of interest, as well as a cropped image of that region. Return JSON output with the following format and all fields completed: {\"app\": \"<what app is the zommed in region focused on>\", \"rich_description\": \"<describe the zoomed-in region including any visual elements>\", \"full_ocr\": \"<all extracted text from the cropped image>\"}"
            )
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error from Gemini API: {e}")
        return None