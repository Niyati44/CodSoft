#!/usr/bin/env python3
"""
Instagram Caption Generator

This script processes flyer images using OCR and generates Instagram captions
using a Mistral language model. It extracts text from images, filters by confidence,
and creates engaging social media captions with hashtags.

Usage:
    python instagram_caption_generator.py <path_to_image>

Requirements:
    - llama-cpp-python
    - Custom OCR module (ocr.py)
    - Mistral model file: mistral-7b-instruct-v0.2.Q4_K_M.gguf
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import contextlib
import re
import logging

try:
    from llama_cpp import Llama
except ImportError:
    print("Error: llama-cpp-python not installed. Install with: pip install llama-cpp-python")
    sys.exit(1)

try:
    from ocr import process_image
except ImportError:
    print("Error: OCR module not found. Please ensure ocr.py is in the same directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def validate_image_path(image_path: str) -> bool:
    """Validate that the image path exists and is a valid image file."""
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    file_ext = Path(image_path).suffix.lower()
    
    if file_ext not in valid_extensions:
        logger.error(f"Invalid image format: {file_ext}. Supported: {', '.join(valid_extensions)}")
        return False
    
    return True


def save_ocr_results(image_path: str, output_file: str = "ocr_output.json") -> Optional[str]:
    """
    Run OCR on an image and save results to a JSON file.
    
    Args:
        image_path: Path to the input image
        output_file: Path to save OCR results
        
    Returns:
        Path to the output file if successful, None otherwise
    """
    try:
        logger.info(f"Processing OCR for image: {image_path}")
        results = process_image(image_path)
        
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"OCR results saved to {output_file}")
        return output_file
    
    except Exception as e:
        logger.error(f"OCR processing failed: {e}")
        return None


def enforce_hashtag_limit(text: str, max_tags: int = 7) -> str:
    """
    Enforce hashtag limit by keeping only the first max_tags hashtags.
    
    Args:
        text: Input text with hashtags
        max_tags: Maximum number of hashtags to keep
        
    Returns:
        Text with limited hashtags
    """
    hashtags = re.findall(r"#\w+", text)
    if len(hashtags) <= max_tags:
        return text

    # Keep only first `max_tags` and remove extras
    allowed_tags = hashtags[:max_tags]
    pattern = r"#\w+"
    
    def replacer(match):
        tag = match.group(0)
        return tag if tag in allowed_tags else ""

    # Clean and reassemble
    cleaned = re.sub(pattern, replacer, text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def extract_high_confidence_text(ocr_data: List[Dict[str, Any]], 
                                confidence_threshold: float = 0.90) -> str:
    """
    Extract high-confidence text from OCR results.
    
    Args:
        ocr_data: List of OCR result dictionaries
        confidence_threshold: Minimum confidence score to include text
        
    Returns:
        Concatenated high-confidence text
    """
    filtered_text = []
    
    for entry in ocr_data:
        if not isinstance(entry, dict):
            continue
            
        confidence = entry.get("confidence", 0)
        text = entry.get("text", "").strip()
        
        # Skip low confidence, empty, or single character entries
        if (confidence >= confidence_threshold and 
            text and 
            len(text) > 1 and 
            not set(text) <= {"-", "_", " "}):
            filtered_text.append(text)
    
    result = " ".join(filtered_text)
    logger.info(f"Extracted {len(filtered_text)} high-confidence text segments")
    return result


def generate_instagram_caption(flyer_text: str, model: Llama) -> str:
    """
    Generate Instagram caption using Mistral model.
    
    Args:
        flyer_text: Extracted text from the flyer
        model: Loaded Llama model instance
        
    Returns:
        Generated Instagram caption
    """
    if not flyer_text.strip():
        logger.warning("No text provided for caption generation")
        return "Unable to generate caption: No text extracted from image."

    prompt = f"""### Instruction:
Generate an emotionally engaging Instagram caption (4–5 lines) from the keywords extracted from flyers.

Keywords:
"{flyer_text}"

The caption must:
- Be engaging, emotionally driven, and urgent while adding a question
- Include a strong call-to-action
- Add any URL or social media handle **only if it is explicitly found in the flyer text**
- Do not create or guess any handle or URL
- **Use 5–7 relevant, trending hashtags — no more than 7**
- Do not repeat hashtags or include irrelevant tags
- Do not add any explanation
- Sound like a real social media post
- Be in a flow and well-framed

### Response:
"""

    try:
        logger.info("Generating caption with Mistral model...")
        result = model(prompt, max_tokens=300, temperature=0.7, top_p=0.9)
        
        logger.debug(f"Raw model result: {result}")

        # Extract text from llama-cpp response
        if isinstance(result, dict) and "choices" in result:
            caption_text = result["choices"][0]["text"].strip()
        elif isinstance(result, str):
            caption_text = result.strip()
        else:
            logger.error(f"Unexpected model output format: {type(result)}")
            return "Caption generation failed: Unexpected model output format."
        
        if not caption_text:
            logger.warning("Model returned empty caption")
            return "Caption generation failed: Empty response from model."
        
        # Enforce hashtag limit
        final_caption = enforce_hashtag_limit(caption_text)
        logger.info("Caption generated successfully")
        return final_caption
        
    except Exception as e:
        logger.error(f"Caption generation error: {e}")
        return f"Caption generation failed: {str(e)}"


def save_caption_to_file(caption: str, image_path: str) -> str:
    """
    Save generated caption to a text file named after the image.
    
    Args:
        caption: Generated caption text
        image_path: Original image path
        
    Returns:
        Path to the saved caption file
    """
    base_name = Path(image_path).stem
    captions_dir = Path("captions")
    captions_dir.mkdir(exist_ok=True)
    
    output_file = captions_dir / f"{base_name}.txt"
    
    try:
        with open(output_file, "w", encoding='utf-8') as f:
            f.write(caption)
        logger.info(f"Caption saved to {output_file}")
        return str(output_file)
    except Exception as e:
        logger.error(f"Failed to save caption: {e}")
        return ""


def load_model(model_path: str = "./mistral-7b-instruct-v0.2.Q4_K_M.gguf") -> Optional[Llama]:
    """
    Load the Mistral model with error handling.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model instance or None if failed
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        logger.info(f"Loading model from {model_path}...")
        with suppress_stdout_stderr():
            llm = Llama(
                model_path=model_path,
                n_ctx=2048,
                n_threads=8,
                verbose=False
            )
        logger.info("Model loaded successfully")
        return llm
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


def main():
    """Main function to orchestrate the caption generation process."""
    if len(sys.argv) != 2:
        print("Usage: python instagram_caption_generator.py <path_to_image>")
        print("\nExample:")
        print("  python instagram_caption_generator.py flyer.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Validate input
    if not validate_image_path(image_path):
        sys.exit(1)
    
    # Process OCR
    ocr_output_path = save_ocr_results(image_path)
    if not ocr_output_path:
        logger.error("OCR processing failed")
        sys.exit(1)
    
    # Load OCR results
    try:
        with open(ocr_output_path, 'r', encoding='utf-8') as f:
            ocr_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OCR results: {e}")
        sys.exit(1)
    
    # Extract high-confidence text
    flyer_text = extract_high_confidence_text(ocr_data)
    if not flyer_text.strip():
        logger.warning("No high-confidence text extracted from image")
        print("Warning: No readable text found in the image. Caption generation may be limited.")
    
    logger.info(f"Extracted text: {flyer_text[:100]}...")
    
    # Load model
    llm = load_model()
    if not llm:
        logger.error("Failed to load language model")
        sys.exit(1)
    
    # Generate caption
    caption = generate_instagram_caption(flyer_text, llm)
    
    # Display results
    print("\n" + "="*50)
    print("GENERATED INSTAGRAM CAPTION")
    print("="*50)
    print(caption)
    print("="*50)
    
    # Save caption
    saved_path = save_caption_to_file(caption, image_path)
    if saved_path:
        print(f"\nCaption saved to: {saved_path}")
    
    # Clean up OCR output file (optional)
    try:
        os.remove(ocr_output_path)
        logger.info("Cleaned up temporary OCR file")
    except Exception:
        pass  # Non-critical error


if __name__ == "__main__":
    main()