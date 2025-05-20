# PigDoc - AI-Powered Pig Disease Detection System

PigDoc is a web application that uses machine learning to detect diseases in pigs through image analysis. The application uses a pre-trained TensorFlow Lite model to analyze uploaded images and identify potential health conditions.

## Features

- Drag and drop image upload
- Real-time image preview
- AI-powered disease detection
- Confidence score for predictions
- Modern, responsive user interface
- Support for multiple image formats

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser with JavaScript enabled

## Installation

1. Clone this repository or download the source code.

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure the TensorFlow Lite model file (`detect.tflite`) is in the correct location.

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Use the application by either:
   - Dragging and dropping an image onto the upload area
   - Clicking "Browse Files" to select an image from your computer

4. Click "Analyze Image" to process the image and view the results

## Supported Disease Categories

- Healthy
- Respiratory Disease
- Skin Disease
- Gastrointestinal Disease

## Technical Details

- Frontend: HTML5, JavaScript, TailwindCSS
- Backend: Flask (Python)
- ML Model: TensorFlow Lite
- Image Processing: OpenCV, Pillow

## Notes

- The application works best with clear, well-lit images of pigs
- Supported image formats: JPG, PNG, JPEG
- For optimal results, ensure the pig is clearly visible in the image

## License

This project is licensed under the MIT License - see the LICENSE file for details. 