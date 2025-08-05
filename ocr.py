#!/usr/bin/env python3
"""
OCR Module for Instagram Caption Generator

This module provides OCR functionality using either Tesseract or PaddleOCR
to extract text from images with confidence scores.

PaddleOCR is preferred as it provides excellent accuracy for various text layouts,
supports multiple languages, and handles rotated text well.
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
PADDLEOCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR available")
except ImportError:
    logger.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    logger.info("PaddleOCR available")
except ImportError:
    logger.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

if not TESSERACT_AVAILABLE and not PADDLEOCR_AVAILABLE:
    print("Error: No OCR library available. Install either pytesseract or paddleocr")
    sys.exit(1)


class OCRProcessor:
    """OCR processor that can use either Tesseract or PaddleOCR."""
    
    def __init__(self, prefer_paddleocr: bool = True):
        """
        Initialize OCR processor.
        
        Args:
            prefer_paddleocr: Whether to prefer PaddleOCR over Tesseract when both are available
        """
        self.prefer_paddleocr = prefer_paddleocr
        self.paddleocr_reader = None
        
        if PADDLEOCR_AVAILABLE and prefer_paddleocr:
            try:
                # Initialize PaddleOCR with English language support
                # use_angle_cls=True helps with rotated text
                # use_gpu=False for CPU inference (set to True if you have GPU)
                self.paddleocr_reader = PaddleOCR(
                    use_angle_cls=True, 
                    lang='en',
                    use_gpu=False,
                    show_log=False  # Suppress PaddleOCR logs
                )
                logger.info("PaddleOCR reader initialized")
            except Exception as e:
                logger.error(f"Failed to initialize PaddleOCR: {e}")
                self.paddleocr_reader = None
    
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
    
    def process_with_paddleocr(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Process image using PaddleOCR.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of dictionaries with text and confidence scores
        """
        if not PADDLEOCR_AVAILABLE or self.paddleocr_reader is None:
            raise RuntimeError("PaddleOCR not available")
        
        try:
            # PaddleOCR returns results in format: [[[bbox], (text, confidence)], ...]
            results_raw = self.paddleocr_reader.ocr(image_path, cls=True)
            
            results = []
            # PaddleOCR can return None for images with no text
            if results_raw and results_raw[0]:
                for detection in results_raw[0]:
                    if detection and len(detection) >= 2:
                        bbox_points = detection[0]  # 4 corner points
                        text_info = detection[1]    # (text, confidence)
                        
                        if text_info and len(text_info) >= 2:
                            text, confidence = text_info
                            
                            if text and text.strip():  # Only include non-empty text
                                # Convert bbox points to x, y, width, height format
                                x_coords = [point[0] for point in bbox_points]
                                y_coords = [point[1] for point in bbox_points]
                                
                                x = min(x_coords)
                                y = min(y_coords)
                                width = max(x_coords) - x
                                height = max(y_coords) - y
                                
                                results.append({
                                    'text': text.strip(),
                                    'confidence': float(confidence),
                                    'bbox': {
                                        'x': int(x),
                                        'y': int(y),
                                        'width': int(width),
                                        'height': int(height)
                                    }
                                })
            
            logger.info(f"PaddleOCR extracted {len(results)} text elements")
            return results
            
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
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
        
        # Try PaddleOCR first if preferred and available
        if self.prefer_paddleocr and PADDLEOCR_AVAILABLE and self.paddleocr_reader:
            try:
                return self.process_with_paddleocr(image_path)
            except Exception as e:
                logger.warning(f"PaddleOCR failed, falling back to Tesseract: {e}")
        
        # Fall back to Tesseract
        if TESSERACT_AVAILABLE:
            try:
                return self.process_with_tesseract(image_path)
            except Exception as e:
                logger.error(f"Tesseract also failed: {e}")
        
        # If PaddleOCR wasn't preferred, try it as backup
        if not self.prefer_paddleocr and PADDLEOCR_AVAILABLE and self.paddleocr_reader:
            try:
                return self.process_with_paddleocr(image_path)
            except Exception as e:
                logger.error(f"PaddleOCR backup also failed: {e}")
        
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