# Instagram Caption Generator

An AI-powered tool that automatically generates engaging Instagram captions from flyer images using OCR (Optical Character Recognition) and the Mistral language model.

## Features

- **Advanced OCR Processing**: Supports both Tesseract and EasyOCR for robust text extraction
- **AI-Powered Caption Generation**: Uses Mistral 7B model to create engaging, contextual captions
- **Smart Hashtag Management**: Automatically limits hashtags to 5-7 relevant tags
- **High-Confidence Text Filtering**: Only uses text with confidence scores above 90%
- **Comprehensive Error Handling**: Robust error handling and logging throughout
- **Multiple Output Formats**: Saves results to both console and text files
- **Image Format Support**: Works with JPG, PNG, BMP, TIFF, and WebP formats

## Requirements

### System Dependencies

For **Tesseract OCR** (recommended):
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Python Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Model File

Download the Mistral model file:
```bash
# Download the Mistral 7B Instruct model (Q4_K_M quantized version)
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

Place the model file in the same directory as the script, or update the `model_path` in the script.

## Installation

1. **Clone or download the project files**:
   ```bash
   # Ensure you have these files:
   # - instagram_caption_generator.py
   # - ocr.py
   # - requirements.txt
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system OCR dependencies** (see Requirements section above)

4. **Download the Mistral model** (see Requirements section above)

## Usage

### Basic Usage

```bash
python instagram_caption_generator.py path/to/your/flyer.jpg
```

### Example

```bash
python instagram_caption_generator.py event_flyer.png
```

### Output

The script will:
1. Extract text from the image using OCR
2. Filter high-confidence text (>90% confidence)
3. Generate an engaging Instagram caption
4. Display the caption in the terminal
5. Save the caption to `captions/[image_name].txt`

### Sample Output

```
==================================================
GENERATED INSTAGRAM CAPTION
==================================================
🎉 Ready to experience the ultimate summer festival? 
Join us for an unforgettable night of music, food, and fun! 
Don't miss out on early bird tickets - they're selling fast! 
What's your favorite festival memory? 

#SummerFestival #MusicFestival #LiveMusic #FoodTrucks #EarlyBird #Festival2024 #GoodVibes
==================================================

Caption saved to: captions/event_flyer.txt
```

## Project Structure

```
instagram-caption-generator/
├── instagram_caption_generator.py  # Main script
├── ocr.py                         # OCR processing module
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── captions/                      # Generated captions (auto-created)
│   └── [image_name].txt
└── mistral-7b-instruct-v0.2.Q4_K_M.gguf  # Model file (download separately)
```

## Configuration

### OCR Settings

You can modify OCR behavior in `ocr.py`:

```python
# Prefer EasyOCR over Tesseract
processor = OCRProcessor(prefer_easyocr=True)

# Or prefer Tesseract
processor = OCRProcessor(prefer_easyocr=False)
```

### Caption Generation Settings

Modify these parameters in `instagram_caption_generator.py`:

```python
# Confidence threshold for text extraction (0.0 to 1.0)
confidence_threshold = 0.90

# Maximum number of hashtags
max_hashtags = 7

# Model parameters
model_params = {
    'max_tokens': 300,
    'temperature': 0.7,
    'top_p': 0.9
}
```

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Ensure the Mistral model file is in the correct location
   - Check the file name matches exactly: `mistral-7b-instruct-v0.2.Q4_K_M.gguf`

2. **"No OCR library available"**
   - Install either `pytesseract` or `easyocr`
   - For Tesseract, also install system dependencies

3. **"No text detected in image"**
   - Check image quality and resolution
   - Ensure text is clearly visible and not too small
   - Try with a different image format

4. **Memory issues with the model**
   - The model requires ~4GB RAM
   - Close other applications to free memory
   - Consider using a smaller model variant

### Debugging

Enable debug logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Performance Tips

1. **OCR Performance**:
   - EasyOCR generally provides better accuracy for complex layouts
   - Tesseract is faster but may struggle with decorative fonts
   - Higher resolution images generally yield better OCR results

2. **Model Performance**:
   - The script loads the model once and reuses it
   - First run may be slower due to model loading
   - Consider using GPU acceleration if available

## Contributing

Feel free to contribute improvements:

1. **OCR Enhancements**: Add support for more OCR engines
2. **Caption Quality**: Improve prompt engineering for better captions
3. **Error Handling**: Add more robust error handling
4. **Performance**: Optimize model loading and inference

## License

This project is open source. Please ensure you comply with the licenses of all dependencies:
- Mistral model: Apache 2.0 License
- OCR libraries: Various open source licenses
- Other dependencies: See individual package licenses

## Acknowledgments

- **Mistral AI** for the language model
- **Tesseract** and **EasyOCR** teams for OCR capabilities
- **llama-cpp-python** for efficient model inference

