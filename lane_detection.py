
from ultralytics import YOLO
import cv2

# Load model YOLOv8 Instance Segmentation
model = YOLO("yolov8n-seg.pt")

def detect_rail_lane(image_path):
    """Mendeteksi jalur rel menggunakan YOLOv8 Instance Segmentation"""
    results = model(image_path, show=True)
    results[0].save("lane_detection_result.jpg")

# Contoh penggunaan
detect_rail_lane(r"D:\Kontrol Cerdas\week 6\rail_segmentation\valid\images\1000195092_0005-0_jpeg.rf.00e151ef8c1d507805f6382d515299b8.jpg")