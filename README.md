# Visual Language Model Webcam to Live Captions

This project uses OpenCV and Hugging Face’s BLIP VLM model to continuously capture frames, run lightweight vision-language inference on CPU or GPU, and overlay natural-language descriptions(live captions), detected objects, and actions in video.

## Project Layout

```
main.py
camera/
  webcam.py
vision/
  model.py
  inference.py
utils/
  config.py
  fps.py
requirements.txt
README.md
```

## Prerequisites

- Python 3.10
- Webcam accessible by OpenCV

## Installation and Running the project

```bash
brew install python@3.10     #if not already installed
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python main.py
```
