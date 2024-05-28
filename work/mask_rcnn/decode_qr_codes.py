#!/usr/bin/env python3

import numpy as np
# import pyboof as pb
import pandas as pd
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import os
from PIL import Image
import argparse

from qreader import QReader

qreader = QReader()

# ----------------- Decoding functions for each QR library ----------------

# PyZBar
def decode_pyzbar(image_path):
    try:
        image = cv2.imread(image_path)
        decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE])
        return decoded_objects[0].data.decode('utf-8') if decoded_objects else None
    except Exception as e:
        print('pyzbar', e)
        

# QReader
def decode_qreader(image_path):
    try:
        image = cv2.imread(image_path)
        decoded_text = qreader.detect_and_decode(image)

        if len(decoded_text):
            return decoded_text[0]
    except Exception as e:
        print('qreader:', e)

# PYBOOF
# https://github.com/lessthanoptimal/PyBoof/tree/SNAPSHOT
def decode_pyboof(image_path):
    try:
        detector = pb.FactoryFiducial(np.uint8).qrcode()
        image = pb.load_single_band(image_path, np.uint8)

        detector.detect(image)

        for qr in detector.detections:
            print(qr.message)
    except Exception as e:
        return None

# Function to decode using OpenCV
def decode_opencv(image_path):
    try:
        image = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(image)
        return data if data else None
    except Exception as e:
        return None


# ----------------- Read QR codes from directory ----------------
    
# Set up argument parser
parser = argparse.ArgumentParser(description='Decode QR codes from images in a directory.')
parser.add_argument('image_folder', type=str, help='Path to the directory containing images')

args = parser.parse_args()

# List to store the results
results = []

# Loop through each image in the folder
for filename in os.listdir(args.image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(args.image_folder, filename)
        result_pyzbar = decode_pyzbar(image_path)
        result_opencv = decode_opencv(image_path)
        result_qreader = decode_qreader(image_path)
        
        print('Pyzbar: ', result_pyzbar)
        print('OpenCV: ', result_opencv)
        print('QReader: ', result_qreader)
        
        results.append({
            'original_image_name': os.path.basename(image_path),  # Get the file name only
            'pyzbar_result': result_pyzbar,
            'opencv_result': result_opencv,
            'qreader_result': result_qreader,
        })

# Create a DataFrame
df = pd.DataFrame(results)

# Calculate metrics
total_images = len(df)
pyzbar_success = df['pyzbar_result'].notna().sum()
opencv_success = df['opencv_result'].notna().sum()
qreader_success = df['qreader_result'].notna().sum()


metrics = {
    'total_images': total_images,
    'pyzbar_success': pyzbar_success,
    'pyzbar_success_percent': (pyzbar_success / total_images) * 100,
    'opencv_success': opencv_success,
    'opencv_success_percent': (opencv_success / total_images) * 100,
    'qreader_success': qreader_success,
    'qreader_success_percent': (qreader_success / total_images) * 100
}

# Print metrics
print(metrics)

# Export DataFrame to CSV
df.to_csv('qr_code_results.csv', index=False)
    
    
    
    