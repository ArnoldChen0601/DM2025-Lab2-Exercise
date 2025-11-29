# Bonus 2: Image Segmentation Comparison
# Compare gemini-2.5-flash-lite vs gemini-2.5-flash models

from PIL import Image, ImageDraw
import io
import base64
import json
import numpy as np
import os
from pydantic import BaseModel
from google.genai import types
import random

# Schema for detected elements
class Element(BaseModel):
    label: str
    box_2d: str
    mask: str

def parse_json(json_output: str):
    """Remove markdown fencing from JSON output"""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])
            json_output = json_output.split("```")[0]
            break
    return json_output

def extract_segmentation_masks(image_path: str, items_to_detect: str, schema, model_name: str, output_dir: str):
    """Extract segmentation masks using specified Gemini model"""

    # High contrast colors for visualization
    HIGH_CONTRAST_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (255, 165, 0), (0, 128, 128),
        (128, 0, 128), (255, 192, 203), (0, 250, 154), (255, 215, 0)
    ]

    # Load and resize image
    im = Image.open(image_path)
    im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)

    # Prompt for segmentation
    prompt = f"""
    Give the segmentation masks for the {items_to_detect}.
    Output a JSON list of segmentation masks where each entry contains the 2D
    bounding box in the key "box_2d", the segmentation mask in key "mask", and
    the text label in the key "label". Use descriptive labels.
    """

    # Configure generation with zero thinking budget for better object detection
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        thinking_config=types.ThinkingConfig(thinking_budget=0)
    )

    input_prompt = [prompt, im]
    response = prompt_gemini(input_prompt=input_prompt, model_name=model_name, schema=schema, new_config=config)

    # Parse JSON response
    items = json.loads(parse_json(response))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create overlay image for visualization
    final_result_image = im.convert("RGBA")

    # Process each detected object
    for i, item in enumerate(items):
        # Extract bounding box coordinates (normalized to 1000)
        box = item["box_2d"]
        y0 = int(box[0] / 1000 * im.size[1])
        x0 = int(box[1] / 1000 * im.size[0])
        y1 = int(box[2] / 1000 * im.size[1])
        x1 = int(box[3] / 1000 * im.size[0])

        # Skip invalid boxes
        if y0 >= y1 or x0 >= x1:
            continue

        # Decode base64 mask
        png_str = item["mask"]
        if not png_str.startswith("data:image/png;base64,"):
            continue

        png_str = png_str.removeprefix("data:image/png;base64,")
        mask_data = base64.b64decode(png_str)
        mask = Image.open(io.BytesIO(mask_data))

        # Resize mask to bounding box dimensions
        mask = mask.resize((x1 - x0, y1 - y0), Image.Resampling.BILINEAR)
        mask_array = np.array(mask)

        # Select random color for visualization
        random_color = random.choice(HIGH_CONTRAST_COLORS)

        # Create mask overlay with transparency
        mask_overlay = Image.new('RGBA', final_result_image.size, (0, 0, 0, 0))
        mask_draw = ImageDraw.Draw(mask_overlay)

        # Draw mask pixels with threshold
        for y in range(y1 - y0):
            for x in range(x1 - x0):
                if mask_array[y, x] > 128:
                    mask_draw.point((x + x0, y + y0), fill=(*random_color, 150))

        # Composite mask onto result
        final_result_image = Image.alpha_composite(final_result_image, mask_overlay)

        # Draw bounding box and label
        final_draw = ImageDraw.Draw(final_result_image)
        final_draw.rectangle([x0, y0, x1, y1], outline=random_color, width=3)
        final_draw.text((x0, y0 - 15), item['label'], fill=random_color)

        # Save individual mask
        mask_filename = f"{item['label']}_{i}_mask.png"
        mask.save(os.path.join(output_dir, mask_filename))

    # Save final result
    final_image_path = os.path.join(output_dir, f"final_result_{model_name.replace('.', '_')}.png")
    final_result_image.save(final_image_path)

    return items, final_image_path

# Main execution
image_path = "./pics/Taipei street scene.png"
items_to_detect = "vehicles"

print("=" * 80)
print("Bonus 2: Segmentation Model Comparison")
print("=" * 80)
print(f"Image: {image_path}")
print(f"Detecting: {items_to_detect}")
print("=" * 80)

# Test with gemini-2.5-flash-lite
print("\n[1] Testing gemini-2.5-flash-lite...")
lite_items, lite_path = extract_segmentation_masks(
    image_path,
    items_to_detect,
    schema=list[Element],
    model_name="gemini-2.5-flash-lite",
    output_dir="segmentation_lite"
)
print(f"✓ Detected {len(lite_items)} objects")
print(f"✓ Result saved to: {lite_path}")

# Test with gemini-2.5-flash
print("\n[2] Testing gemini-2.5-flash...")
flash_items, flash_path = extract_segmentation_masks(
    image_path,
    items_to_detect,
    schema=list[Element],
    model_name="gemini-2.5-flash",
    output_dir="segmentation_flash"
)
print(f"✓ Detected {len(flash_items)} objects")
print(f"✓ Result saved to: {flash_path}")

# Comparison and Discussion
print("\n" + "=" * 80)
print("DISCUSSION AND COMPARISON")
print("=" * 80)

print(f"\n[Detection Count]")
print(f"  - flash-lite: {len(lite_items)} objects")
print(f"  - flash:      {len(flash_items)} objects")

print(f"\n[Detected Labels]")
print(f"  flash-lite: {[item['label'] for item in lite_items]}")
print(f"  flash:      {[item['label'] for item in flash_items]}")

print(f"\n[Model Comparison]")
print(f"  1. Detection Accuracy:")
print(f"     - flash-lite is optimized for speed with lower computational cost")
print(f"     - flash provides higher accuracy with better feature recognition")

print(f"\n  2. Segmentation Quality:")
print(f"     - flash-lite may produce coarser segmentation masks")
print(f"     - flash generates finer-grained masks with better edge precision")

print(f"\n  3. Use Cases:")
print(f"     - flash-lite: Real-time applications, batch processing, cost-sensitive tasks")
print(f"     - flash: High-precision requirements, detailed analysis, production systems")

print(f"\n  4. Performance Trade-offs:")
print(f"     - flash-lite: Faster inference, lower token usage, reduced API costs")
print(f"     - flash: Better semantic understanding, more reliable object detection")

print("\n" + "=" * 80)
