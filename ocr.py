#!/usr/bin/env python3
"""
OCR Module for Instagram Caption Generator

This module provides OCR functionality using either Tesseract or EasyOCR
to extract text from images with confidence scores.
"""

import os
import sys
from typing import List, Dict, Any, Optional
import logging

try:
    from PIL import Image
    import numpy as np
except ImportError:
    print("Error: PIL and numpy are required. Install with: pip install Pillow numpy")
    sys.exit(1)

# Configure logging
logger = logging.getLogger(__name__)

# Try to import OCR libraries
TESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR available")
except ImportError:
    logger.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    logger.info("EasyOCR available")
except ImportError:
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

if not TESSERACT_AVAILABLE and not EASYOCR_AVAILABLE:
    print("Error: No OCR library available. Install either pytesseract or easyocr")
    sys.exit(1)


class OCRProcessor:
    """OCR processor that can use either Tesseract or EasyOCR."""
    
    def __init__(self, prefer_easyocr: bool = True):
        """
        Initialize OCR processor.
        
        Args:
            prefer_easyocr: Whether to prefer EasyOCR over Tesseract when both are available
        """
        self.prefer_easyocr = prefer_easyocr
        self.easyocr_reader = None
        
        if EASYOCR_AVAILABLE and prefer_easyocr:
            try:
                self.easyocr_reader = easyocr.Reader(['en'])
                logger.info("EasyOCR reader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize EasyOCR: {e}")
                self.easyocr_reader = None
    
    def process_with_tesseract(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process image using Tesseract OCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and confidence scores
        """
        if not TESSERACT_AVAILABLE:
            raise RuntimeError("Tesseract not available")
        
        try:
            image = Image.open(image_path)
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            results = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > 0 and text:  # Only include confident detections with text
                    results.append({
                        'text': text,
                        'confidence': confidence / 100.0,  # Convert to 0-1 scale
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        }
                    })
            
            logger.info(f"Tesseract extracted {len(results)} text elements")
            return results
            
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")
            return []
    
    def process_with_easyocr(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process image using EasyOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and confidence scores
        """
        if not EASYOCR_AVAILABLE or self.easyocr_reader is None:
            raise RuntimeError("EasyOCR not available")
        
        try:
            results_raw = self.easyocr_reader.readtext(image_path)
            
            results = []
            for detection in results_raw:
                bbox_points, text, confidence = detection
                
                if text.strip():  # Only include non-empty text
                    # Convert bbox points to x, y, width, height format
                    x_coords = [point[0] for point in bbox_points]
                    y_coords = [point[1] for point in bbox_points]
                    
                    x = min(x_coords)
                    y = min(y_coords)
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    results.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': {
                            'x': int(x),
                            'y': int(y),
                            'width': int(width),
                            'height': int(height)
                        }
                    })
            
            logger.info(f"EasyOCR extracted {len(results)} text elements")
            return results
            
        except Exception as e:
            logger.error(f"EasyOCR failed: {e}")
            return []
    
    def process_image(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process image using the best available OCR method.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and confidence scores
        """
        if not os.path.exists(image_path):
            logger.error(f"Image file not found: {image_path}")
            return []
        
        # Try EasyOCR first if preferred and available
        if self.prefer_easyocr and EASYOCR_AVAILABLE and self.easyocr_reader:
            try:
                return self.process_with_easyocr(image_path)
            except Exception as e:
                logger.warning(f"EasyOCR failed, falling back to Tesseract: {e}")
        
        # Fall back to Tesseract
        if TESSERACT_AVAILABLE:
            try:
                return self.process_with_tesseract(image_path)
            except Exception as e:
                logger.error(f"Tesseract also failed: {e}")
        
        # If EasyOCR wasn't preferred, try it as backup
        if not self.prefer_easyocr and EASYOCR_AVAILABLE and self.easyocr_reader:
            try:
                return self.process_with_easyocr(image_path)
            except Exception as e:
                logger.error(f"EasyOCR backup also failed: {e}")
        
        logger.error("All OCR methods failed")
        return []


# Global OCR processor instance
_ocr_processor = None

def get_ocr_processor() -> OCRProcessor:
    """Get or create the global OCR processor instance."""
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor


def process_image(image_path: str) -> List[Dict[str, Any]]:
    """
    Main function to process an image and extract text with confidence scores.
    
    This is the function that should be imported and used by other modules.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of dictionaries with the following structure:
        [
            {
                'text': 'extracted text',
                'confidence': 0.95,  # confidence score between 0 and 1
                'bbox': {
                    'x': 100,
                    'y': 50,
                    'width': 200,
                    'height': 30
                }
            },
            ...
        ]
    """
    processor = get_ocr_processor()
    return processor.process_image(image_path)


def main():
    """Test the OCR functionality."""
    if len(sys.argv) != 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    results = process_image(image_path)
    
    print(f"\nOCR Results for: {image_path}")
    print("=" * 50)
    
    if not results:
        print("No text detected in the image.")
        return
    
    for i, result in enumerate(results, 1):
        print(f"{i}. Text: '{result['text']}'")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   BBox: {result['bbox']}")
        print()
    
    # Save results to JSON for inspection
    import json
    output_file = "ocr_test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()