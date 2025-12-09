import os
os.environ["FLAGS_use_mkldnn"] = "0"    # bắt buộc phải trước import paddle
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import figure
from PIL import Image
import difflib
import re
import math
import json
import sys
import argparse

import torch

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg

# Specifying output path and font path.
FONT = './PaddleOCR/doc/fonts/latin.ttf'


def predict(recognitor, detector, img_path, padding=4):
    # Load image
    img = cv2.imread(img_path)

    # Text detection
    try:
        result = detector.ocr(img_path, cls=False, det=True, rec=False)
        # normalize nested result formats
        try:
            result = result[:][:][0]
        except Exception:
            pass
    except Exception as e:
        print('Detector failed with error:', e)
        print('Falling back to recognizer-only (whole-image) mode.')
        # fallback: run recognitor on whole image if available
        if recognitor is None:
            return [], []
        try:
            # convert BGR->RGB for PIL
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            rec_result = recognitor.predict(pil)
            text = rec_result if isinstance(rec_result, str) else (rec_result[0] if len(rec_result) > 0 else '')
            print(text)
            return [], [text]
        except Exception as e2:
            print('Recognizer fallback also failed:', e2)
            return [], []

    # Filter Boxes
    boxes = []
    for line in result:
        boxes.append([[int(line[0][0]), int(line[0][1])], [int(line[2][0]), int(line[2][1])]])
    boxes = boxes[::-1]

    # Add padding to boxes
    padding = 4
    for box in boxes:
        box[0][0] = box[0][0] - padding
        box[0][1] = box[0][1] - padding
        box[1][0] = box[1][0] + padding
        box[1][1] = box[1][1] + padding

    # Text recognizion
    texts = []
    for box in boxes:
        cropped_image = img[box[0][1]:box[1][1], box[0][0]:box[1][0]]
        try:
            cropped_image = Image.fromarray(cropped_image)
        except:
            continue

        rec_result = recognitor.predict(cropped_image)
        text = rec_result#[0]

        texts.append(text)
        print(text)

    return boxes, texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', required=True, help='foo help')
    parser.add_argument('--output', default='./runs/predict', help='path to save output file')
    parser.add_argument('--use_gpu', required=False, help='is use GPU?')
    parser.add_argument('--no_detector', action='store_true', help='Skip PaddleOCR detector and use vietocr recognizer only')
    args = parser.parse_args()

    # Configure of VietOCR
    # Default weight
    config = Cfg.load_config_from_name('vgg_transformer')
    # Custom weight
    # config = Cfg.load_config_from_file('vi00_vi01_transformer.yml')
    # config['weights'] = './pretrain_ocr/vi00_vi01_transformer.pth'

    config['cnn']['pretrained'] = True
    config['predictor']['beamsearch'] = True
    config['device'] = 'cpu'

    recognitor = Predictor(config)

    # Config of PaddleOCR (optional)
    detector = None
    if not args.no_detector:
        try:
            # lazy import so module import doesn't require PaddleOCR/paddlepaddle
            from PaddleOCR import PaddleOCR, draw_ocr
            detector = PaddleOCR(use_angle_cls=False, lang="vi", use_gpu=False)
        except Exception as e:
            print('Failed to initialize PaddleOCR detector:', e)
            print('Continuing with recognizer-only mode.')

    # Predict
    boxes, texts = predict(recognitor, detector, args.img, padding=2)

    # Write output lines
    out_path = args.output
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            for t in texts:
                f.write((t or '').strip() + '\n')
        print(f'Wrote {len(texts)} lines to {out_path}')
    except Exception as e:
        print('Failed to write output file:', e)


if __name__ == "__main__":    
    main()