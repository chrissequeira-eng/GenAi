import pytesseract
from PIL import Image

# Local Tesseract path (adjust to where you stored tesseract.exe)
pytesseract.pytesseract.tesseract_cmd = r"Z:\Genai_Projects\teseract\tesseract.exe"

# Open image with raw string
img = Image.open(r"Z:\Genai_Projects\Multimodal_Assistant\Knowledge_Base\Structured outputUsage.png")

# Run OCR
text = pytesseract.image_to_string(img)

print("OCR Result:\n", text)
