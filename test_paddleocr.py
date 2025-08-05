#!/usr/bin/env python3
"""
Test script for PaddleOCR integration

This script tests the PaddleOCR functionality with a sample image
to ensure everything is working correctly.
"""

import sys
import json
from pathlib import Path

try:
    from ocr import process_image
    print("✓ OCR module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import OCR module: {e}")
    sys.exit(1)

def test_paddleocr():
    """Test PaddleOCR with sample text or image."""
    print("\nTesting PaddleOCR functionality...")
    
    # Check if we have any image files to test with
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    test_images = []
    
    for ext in image_extensions:
        test_images.extend(Path('.').glob(f'*{ext}'))
        test_images.extend(Path('.').glob(f'*{ext.upper()}'))
    
    if not test_images:
        print("No test images found in current directory.")
        print("Please provide an image file to test:")
        print("  python test_paddleocr.py <image_path>")
        return False
    
    # Use the first found image
    test_image = str(test_images[0])
    print(f"Testing with image: {test_image}")
    
    try:
        results = process_image(test_image)
        
        if results:
            print(f"\n✓ Successfully extracted {len(results)} text elements:")
            for i, result in enumerate(results[:5], 1):  # Show first 5 results
                print(f"  {i}. '{result['text'][:50]}...' (confidence: {result['confidence']:.2f})")
            
            if len(results) > 5:
                print(f"  ... and {len(results) - 5} more")
            
            # Save results for inspection
            with open('test_ocr_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Full results saved to: test_ocr_results.json")
            return True
        else:
            print("✗ No text detected in the image")
            return False
            
    except Exception as e:
        print(f"✗ OCR processing failed: {e}")
        return False

def main():
    """Main test function."""
    print("PaddleOCR Integration Test")
    print("=" * 30)
    
    if len(sys.argv) > 1:
        # Test with provided image
        image_path = sys.argv[1]
        if not Path(image_path).exists():
            print(f"✗ Image file not found: {image_path}")
            sys.exit(1)
        
        print(f"Testing with provided image: {image_path}")
        
        try:
            results = process_image(image_path)
            
            if results:
                print(f"\n✓ Successfully extracted {len(results)} text elements:")
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Text: '{result['text']}'")
                    print(f"     Confidence: {result['confidence']:.3f}")
                    print(f"     BBox: {result['bbox']}")
                    print()
                
                # Save results
                output_file = f"test_results_{Path(image_path).stem}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"✓ Results saved to: {output_file}")
            else:
                print("✗ No text detected in the image")
                
        except Exception as e:
            print(f"✗ OCR processing failed: {e}")
            sys.exit(1)
    else:
        # Auto-test with available images
        success = test_paddleocr()
        if not success:
            sys.exit(1)
    
    print("\n🎉 PaddleOCR test completed successfully!")

if __name__ == "__main__":
    main()