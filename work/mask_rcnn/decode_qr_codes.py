#!/usr/bin/env python3

import numpy as np
import pyboof as pb
import pandas as pd
from pyzbar.pyzbar import decode, ZBarSymbol
import cv2
import os
from PIL import Image
import argparse
from dbr import *

from qreader import QReader

import seaborn as sns
import matplotlib.pyplot as plt


# Initialize QReader
qreader = QReader()

# Initialize Dynamsoft Reader with license key (30 day free trial available)
BarcodeReader.init_license("")
dbr_reader = BarcodeReader()


# ----------------- Decoding functions for each QR library ----------------

# Dynamsoft barcode reader: https://www.dynamsoft.com/barcode-reader/barcode-types/qr-code/
def decode_dbr(image_path):
    try:
        results = dbr_reader.decode_file(image_path)
        if results is not None:
            for text_result in results:
                return text_result.barcode_text
    except Exception as e:
        print('dbr error:', e)
        return None
    
# PyZBar: https://github.com/NaturalHistoryMuseum/pyzbar/
def decode_pyzbar(image_path):
    try:
        image = cv2.imread(image_path)
        decoded_objects = decode(image, symbols=[ZBarSymbol.QRCODE])
        return decoded_objects[0].data.decode('utf-8') if decoded_objects else None
    except Exception as e:
        print('pyzbar error:', e)
        return None

# QReader: https://github.com/Eric-Canas/qreader
def decode_qreader(image_path):
    try:
        image = cv2.imread(image_path)
        decoded_text = qreader.detect_and_decode(image)

        if len(decoded_text):
            return decoded_text[0]
    except Exception as e:
        print('qreader error:', e)
        return None

# Pyboof: https://github.com/lessthanoptimal/PyBoof/tree/SNAPSHOT
def decode_pyboof(image_path):
    try:
        detector = pb.FactoryFiducial(np.uint8).qrcode()
        image = pb.load_single_band(image_path, np.uint8)

        detector.detect(image)

        for qr in detector.detections:
            return qr.message
    except Exception as e:
        print('pyboof error:', e)
        return None

# OpenCV: 
def decode_opencv(image_path):
    try:
        image = cv2.imread(image_path)
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(image)
        return data if data else None
    except Exception as e:
        print('opencv error:', e)
        return None


# ----------------- Read QR codes from directory ----------------
    
# Set up argument parser
parser = argparse.ArgumentParser(description='Decode QR codes from images in a directory.')
parser.add_argument('image_folder', type=str, help='Path to the directory containing images')

args = parser.parse_args()
image_folder = args.image_folder

decoding_functions = [
    ('pyzbar', decode_pyzbar),
    ('opencv', decode_opencv),
    ('qreader', decode_qreader),
    ('dbr', decode_dbr),
    ('pyboof', decode_pyboof),
]

# List to store the results
results = []

# Loop through each image in the folder
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, filename)
        result = {'original_image_name': os.path.basename(image_path)}
        
        for name, func in decoding_functions:
            result[f'{name}_result'] = func(image_path)
        
        results.append(result)

# Create a DataFrame
df = pd.DataFrame(results)

# Calculate metrics for each library
total_images = len(df)
metrics = {'total_images': total_images}
success_percentages = []

for name, _ in decoding_functions:
    success_count = df[f'{name}_result'].notna().sum()
    success_percent = (success_count / total_images) * 100
    metrics[f'{name}_success_percent'] = success_percent
    success_percentages.append({'library': name, 'success_percent': success_percent})

# Add a column to the DataFrame to indicate success with any library
df['success'] = df[[f'{name}_result' for name, _ in decoding_functions]].notna().any(axis=1)

# Calculate the percentage of successful images
success_percent = (df['success'].sum() / len(df)) * 100

# Print metrics
print(metrics)
print(f"Percentage of images with successful decoding: {success_percent:.2f}%")

# Export DataFrame to CSV
df.to_csv('open_source_qr_code_results.csv', index=False)
    
    
    
    