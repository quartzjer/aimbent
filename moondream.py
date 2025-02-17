import sys
from transformers import AutoModelForCausalLM
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    #device_map={"": "cuda"}
)

def main():
    if len(sys.argv) < 2:
        print("Usage: python moondream.py <image_path>")
        sys.exit(1)
    image_path = sys.argv[1]
    
    image = Image.open(image_path)
    enc_image = model.encode_image(image)
    result = model.query(enc_image, "Describe this image.")
    print(result)
    
if __name__ == "__main__":
    main()
