#!/usr/bin/env python3
"""
app.py - Flask OCR App (OPTIMIZED)
- VietOCR recognizer with caching
- PaddleOCR detector with caching
- Load models once, use many times
- Fast OCR processing
"""

import os
import sys
import uuid
import time
import json
import threading
from pathlib import Path
from text_corrector import correct_ocr_text
import re

# Fix Paddle MKL errors - PHẢI ĐẶT TRƯỚC KHI IMPORT PADDLE
os.environ["FLAGS_use_mkldnn"] = "1"  # BẬT MKLDNN để tăng tốc CPU
os.environ["FLAGS_enable_mkldnn"] = "1"
os.environ["FLAGS_enable_one_dnn"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Quan trọng cho Windows

from flask import Flask, render_template, request, jsonify, send_file
from PIL import Image
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

# VietOCR
from vietocr.vietocr.tool.config import Cfg
from vietocr.vietocr.tool.predictor import Predictor

# Flask path setup
TEMPLATES_FOLDER = str(BASE_DIR / "webapp" / "templates")
STATIC_FOLDER = str(BASE_DIR / "webapp" / "static")
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULT_FILE = str(BASE_DIR / "result.txt")

app = Flask(__name__, template_folder=TEMPLATES_FOLDER, static_folder=STATIC_FOLDER)

# ================================
# MODEL CACHE - Load once, use many times
# ================================
class ModelCache:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.recognizer = None
        self.detector = None
        self.recognizer_loaded = False
        self.detector_loaded = False
        self.recognizer_lock = threading.Lock()
        self.detector_lock = threading.Lock()
        self._initialized = True
    
    def get_recognizer(self):
        """Lấy VietOCR recognizer (lazy loading)"""
        if not self.recognizer_loaded:
            with self.recognizer_lock:
                if not self.recognizer_loaded:
                    print("[CACHE] Loading VietOCR model...")
                    start = time.time()
                    cfg = Cfg.load_config_from_name("vgg_transformer")
                    cfg["cnn"]["pretrained"] = True
                    cfg["predictor"]["beamsearch"] = False  # Tắt beamsearch để tăng tốc
                    cfg["device"] = "cpu"
                    self.recognizer = Predictor(cfg)
                    self.recognizer_loaded = True
                    print(f"[CACHE] VietOCR loaded in {time.time() - start:.2f}s")
        return self.recognizer
    
    def get_detector(self):
        """Lấy PaddleOCR detector (lazy loading)"""
        if not self.detector_loaded:
            with self.detector_lock:
                if not self.detector_loaded:
                    print("[CACHE] Loading PaddleOCR detector...")
                    start = time.time()
                    try:
                        from PaddleOCR import PaddleOCR
                        # Cấu hình tối ưu cho tốc độ
                        self.detector = PaddleOCR(
                            use_angle_cls=False,        # Tắt góc nghiêng để tăng tốc
                            lang="vi",
                            use_gpu=False,             # Không dùng GPU
                            show_log=False,            # Tắt log để tăng tốc
                            enable_mkldnn=True,        # Bật MKLDNN acceleration
                            det_db_thresh=0.3,         # Ngưỡng thấp hơn = nhanh hơn
                            det_db_box_thresh=0.5,
                            det_db_unclip_ratio=1.5,
                            det_limit_side_len=960,    # Giới hạn kích thước ảnh
                            det_limit_type='max',
                            use_dilation=False,        # Tắt dilation để tăng tốc
                            det_model_dir=None,
                            rec_model_dir=None,
                            cls_model_dir=None
                        )
                        self.detector_loaded = True
                        print(f"[CACHE] PaddleOCR loaded in {time.time() - start:.2f}s")
                    except Exception as e:
                        print(f"[CACHE] Failed to load PaddleOCR: {e}")
                        self.detector = None
                        self.detector_loaded = True  # Đánh dấu đã thử load
        return self.detector
    
    def warmup(self):
        """Pre-load models khi server start"""
        print("[CACHE] Pre-loading models in background...")
        
        def load_recognizer():
            self.get_recognizer()
        
        def load_detector():
            self.get_detector()
        
        # Load song song trong background
        t1 = threading.Thread(target=load_recognizer)
        t2 = threading.Thread(target=load_detector)
        t1.start()
        t2.start()
        
        # Chờ 5 giây cho việc load
        t1.join(timeout=5)
        t2.join(timeout=5)
        
        print("[CACHE] Models pre-loaded")

# Tạo global cache
MODEL_CACHE = ModelCache()

# ================================
# OCR Function - OPTIMIZED VERSION
# ================================
def correct_ocr_text(text: str) -> str:
    """Simple Vietnamese text corrector"""
    if not text or not text.strip() or text == "Không tìm thấy văn bản":
        return text
    
    # Fix common OCR errors
    errors = {
        'băng': 'bằng', 'đề': 'để', 'cân': 'cần', 'nam': 'nhằm',
        'thiếu số': 'thiểu số', 'chính phù': 'chính phủ', 'chính phú': 'chính phủ',
        'dê': 'dễ', 'tiếp thụ': 'tiếp thu', 'kiên thức': 'kiến thức',
        'học tạp': 'học tập', 'phố biến': 'phổ biến', 'giao dich': 'giao dịch',
        'quốc tê': 'quốc tế', 'liên tụ': 'liên tục', 'văn hoá': 'văn hóa',
        'bản sác': 'bản sắc', 'phát hui': 'phát huy', 'giữ gin': 'giữ gìn',
    }
    
    for wrong, correct in errors.items():
        text = text.replace(wrong, correct)
    
    # Fix line numbers and formatting
    lines = text.split('\n')
    fixed = []
    for line in lines:
        if line.strip() and line[0].isdigit():
            # "22 " -> "2. "
            line = re.sub(r'^(\d{2,})\s', lambda m: f"{m.group(1)[0]}. ", line)
            # "1" -> "1. "
            line = re.sub(r'^(\d+)(?![\.\d])', r'\1. ', line)
        fixed.append(line)
    
    text = '\n'.join(fixed)
    text = re.sub(r'([a-z,])\n([a-z])', r'\1 \2', text)  # Join broken words
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    text = re.sub(r'\n\s*\n+', '\n\n', text)  # Remove extra blank lines
    
    return text.strip()
def ocr_image_fast(img_path, padding=2):
    """
    Hàm OCR tối ưu tốc độ - sử dụng cached models
    """
    start_time = time.time()
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Cannot load image from {img_path}")
        return [], []
    
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}")
    
    # Lấy models từ cache
    recognizer = MODEL_CACHE.get_recognizer()
    detector = MODEL_CACHE.get_detector()
    
    # Text detection
    boxes = []
    detection_time = 0
    
    if detector is not None:
        try:
            det_start = time.time()
            
            # Giảm kích thước ảnh nếu quá lớn để tăng tốc
            max_size = 1280
            scale = 1.0
            det_img_path = img_path
            
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h))
                temp_path = img_path + "_resized.jpg"
                cv2.imwrite(temp_path, img_resized)
                det_img_path = temp_path
            
            # Chạy detection
            result = detector.ocr(det_img_path, cls=False, det=True, rec=False)
            
            # Xóa file tạm nếu có
            if 'temp_path' in locals() and os.path.exists(det_img_path) and det_img_path != img_path:
                os.remove(det_img_path)
            
            detection_time = time.time() - det_start
            
            # Xử lý kết quả
            if result and isinstance(result, list) and len(result) > 0:
                detection_blocks = result[0]
                
                for block in detection_blocks:
                    if isinstance(block, list) and len(block) >= 4:
                        points = block[:4]
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        
                        # Scale back về kích thước gốc
                        x1 = int(min(xs) / scale) if scale != 1.0 else int(min(xs))
                        y1 = int(min(ys) / scale) if scale != 1.0 else int(min(ys))
                        x2 = int(max(xs) / scale) if scale != 1.0 else int(max(xs))
                        y2 = int(max(ys) / scale) if scale != 1.0 else int(max(ys))
                        
                        boxes.append([[x1, y1], [x2, y2]])
            
            print(f"Detection: {detection_time:.2f}s, Found {len(boxes)} boxes")
            
        except Exception as e:
            print(f'Detection error: {e}')
    
    # Nếu không detect được, dùng toàn ảnh
    if not boxes:
        print("No boxes detected, using whole image")
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(img_rgb)
            text = recognizer.predict(pil) if recognizer else ""
            text = text if isinstance(text, str) else text
            return [], [text]
        except:
            return [], []
    
    # Sắp xếp boxes từ trên xuống dưới
    if boxes:
        boxes_with_centers = []
        for box in boxes:
            x1, y1 = box[0]
            x2, y2 = box[1]
            y_center = (y1 + y2) / 2
            boxes_with_centers.append((box, y_center))
        
        boxes_with_centers.sort(key=lambda x: x[1])
        boxes = [box for box, _ in boxes_with_centers]
    
    # Add padding và clamp tọa độ
    for box in boxes:
        box[0][0] = max(0, box[0][0] - padding)
        box[0][1] = max(0, box[0][1] - padding)
        box[1][0] = min(w-1, box[1][0] + padding)
        box[1][1] = min(h-1, box[1][1] + padding)
    
    # Text recognition
    texts = []
    recognition_start = time.time()
    
    for i, box in enumerate(boxes):
        x1, y1 = box[0]
        x2, y2 = box[1]
        
        if x2 <= x1 or y2 <= y1:
            texts.append("")
            continue
        
        try:
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                texts.append("")
                continue
                
            # Chuyển sang PIL
            crop_pil = Image.fromarray(crop)
            
            # Recognition
            text = recognizer.predict(crop_pil) if recognizer else ""
            text = text if isinstance(text, str) else text
            
            texts.append(text)
            
        except Exception as e:
            texts.append("")
    
    recognition_time = time.time() - recognition_start
    total_time = time.time() - start_time
    
    print(f"Recognition: {recognition_time:.2f}s, Total: {total_time:.2f}s")
    return boxes, texts

# ================================
# Routes
# ================================
@app.route("/")
def index():
    """Main page - serves the web interface"""
    text = ""
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, "r", encoding="utf8") as f:
            text = f.read()
    return render_template("index.html", text=text)

@app.route("/ocr", methods=["POST"])
def ocr():
    """Traditional form-based OCR endpoint"""
    progress = []
    
    # Receive file
    if "image" not in request.files:
        return render_template("index.html", text="Không có file!")
    
    file_storage = request.files["image"]
    if file_storage.filename == "":
        return render_template("index.html", text="Bạn chưa chọn ảnh!")
    
    # Validate và lưu file
    ext = os.path.splitext(file_storage.filename)[1].lower()
    allowed_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    if ext not in allowed_ext:
        return render_template("index.html", text=f"File không hỗ trợ: {ext}")
    
    fname = str(uuid.uuid4()) + ext
    path = os.path.join(UPLOAD_FOLDER, fname)
    file_storage.save(path)
    
    progress.append(f"Đã nhận file: {file_storage.filename}")
    progress.append("Đang xử lý OCR...")
    
    # OCR với hàm tối ưu
    boxes, texts = ocr_image_fast(path, padding=2)
    
    progress.append(f"Đã nhận diện {len([t for t in texts if t])} dòng văn bản")
    
    # Tạo ảnh với bounding boxes
    preview_url = f"/static/uploads/{fname}"
    if boxes and texts:
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                for i, (box, text) in enumerate(zip(boxes, texts)):
                    if text and text.strip():
                        x1, y1 = box[0]
                        x2, y2 = box[1]
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                outname = "annot_" + fname
                outpath = os.path.join(UPLOAD_FOLDER, outname)
                cv2.imwrite(outpath, img_bgr)
                preview_url = f"/static/uploads/{outname}"
        except:
            pass
    
    # Lưu kết quả
    final_text = "\n".join([t.strip() for t in texts if t and t.strip()]) or "Không tìm thấy văn bản"
    try:
        with open(RESULT_FILE, "w", encoding="utf8") as f:
            f.write(final_text)
        progress.append("Đã lưu kết quả")
    except:
        pass
    
    return render_template(
        "index.html", 
        text=final_text, 
        image_url=preview_url,
        progress="\n".join(progress)
    )

@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    """API endpoint for AJAX/JSON requests - OPTIMIZED"""
    start_time = time.time()
    
    # Check file
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image provided"}), 400
    
    file_storage = request.files["image"]
    if file_storage.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    # Validate
    original_filename = file_storage.filename
    ext = os.path.splitext(original_filename)[1].lower()
    allowed_ext = {'.png', '.jpg', '.jpeg', '.bmp'}
    
    if ext not in allowed_ext:
        return jsonify({
            "success": False,
            "error": f"Unsupported file type: {ext}. Allowed: {', '.join(allowed_ext)}"
        }), 400
    
    # Check size
    file_storage.seek(0, os.SEEK_END)
    file_size = file_storage.tell()
    file_storage.seek(0)
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({
            "success": False,
            "error": f"File too large ({file_size/1024/1024:.1f} MB). Max 10MB"
        }), 400
    
    # Save file
    fname = str(uuid.uuid4()) + ext
    path = os.path.join(UPLOAD_FOLDER, fname)
    file_storage.save(path)
    
    # OCR với hàm tối ưu
    ocr_start = time.time()
    try:
        boxes, texts = ocr_image_fast(path, padding=2)
        ocr_time = time.time() - ocr_start
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"OCR processing failed: {str(e)}"
        }), 500
    
    # Tạo annotated image
    preview_url = None
    if boxes and texts:
        try:
            img_bgr = cv2.imread(path)
            if img_bgr is not None:
                for i, (box, text) in enumerate(zip(boxes, texts)):
                    if text and text.strip():
                        x1, y1 = box[0]
                        x2, y2 = box[1]
                        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                outname = "annot_" + fname
                outpath = os.path.join(UPLOAD_FOLDER, outname)
                cv2.imwrite(outpath, img_bgr)
                preview_url = f"/static/uploads/{outname}"
        except:
            pass
    
    # Prepare results
    valid_texts = [t.strip() for t in texts if t and t.strip()]
    final_text = "\n".join(valid_texts) if valid_texts else ""
    
    # Save to result file
    try:
        with open(RESULT_FILE, "w", encoding="utf8") as f:
            f.write(final_text)
    except:
        pass
    
    # Stats
    total_time = time.time() - start_time
    char_count = sum(len(t) for t in valid_texts)
    word_count = sum(len(t.split()) for t in valid_texts)
    line_count = len(valid_texts)
    
    return jsonify({
        "success": True,
        "text": final_text,
        "preview": preview_url,
        "stats": {
            "processing_time": round(total_time, 2),
            "ocr_time": round(ocr_time, 2),
            "boxes_detected": len(boxes),
            "text_lines": line_count,
            "characters": char_count,
            "words": word_count
        },
        "details": {
            "file_name": original_filename,
            "file_size": file_size
        },
        "message": f"OCR completed in {total_time:.1f}s"
    })

@app.route("/api/status", methods=["GET"])
def api_status():
    """API endpoint to check server and model status"""
    recognizer = MODEL_CACHE.get_recognizer()
    detector = MODEL_CACHE.get_detector()
    
    models_status = {
        "vietocr": recognizer is not None,
        "paddleocr": detector is not None,
        "models_cached": MODEL_CACHE.recognizer_loaded and MODEL_CACHE.detector_loaded
    }
    
    # Check uploads folder
    uploads_count = 0
    if os.path.exists(UPLOAD_FOLDER):
        uploads_count = len([f for f in os.listdir(UPLOAD_FOLDER) 
                           if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))])
    
    return jsonify({
        "status": "online",
        "server_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models_status,
        "storage": {
            "uploads_folder": os.path.exists(UPLOAD_FOLDER),
            "uploads_count": uploads_count,
            "result_file": os.path.exists(RESULT_FILE)
        }
    })

@app.route("/api/download", methods=["GET"])
def api_download():
    """Download OCR result as TXT file"""
    if not os.path.exists(RESULT_FILE):
        return jsonify({"success": False, "error": "No result file found"}), 404
    
    with open(RESULT_FILE, "r", encoding="utf8") as f:
        content = f.read()
    
    if not content.strip():
        return jsonify({"success": False, "error": "Result file is empty"}), 404
    
    return send_file(
        RESULT_FILE,
        as_attachment=True,
        download_name=f"ocr_result_{time.strftime('%Y%m%d_%H%M%S')}.txt",
        mimetype='text/plain'
    )

@app.route("/api/clear", methods=["POST"])
def api_clear():
    """Clear results and uploaded files"""
    try:
        # Clear result file
        if os.path.exists(RESULT_FILE):
            os.remove(RESULT_FILE)
        
        # Clear uploads (keep last 5 files)
        if os.path.exists(UPLOAD_FOLDER):
            files = []
            for f in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, f)
                if os.path.isfile(filepath):
                    files.append((filepath, os.path.getmtime(filepath)))
            
            # Sắp xếp theo thời gian (cũ nhất trước)
            files.sort(key=lambda x: x[1])
            
            # Xóa tất cả trừ 5 file mới nhất
            if len(files) > 5:
                for i in range(len(files) - 5):
                    try:
                        os.remove(files[i][0])
                    except:
                        pass
        
        return jsonify({
            "success": True,
            "message": "Cleared results and old uploads"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"success": False, "error": "Internal server error"}), 500

# ================================
# Main execution
# ================================
if __name__ == "__main__":
    print("="*60)
    print("Vietnamese OCR Flask Application - OPTIMIZED")
    print("="*60)
    print(f"📁 Upload folder: {UPLOAD_FOLDER}")
    print(f"📄 Result file: {RESULT_FILE}")
    print("="*60)
    
    # Pre-load models
    print("Pre-loading models...")
    MODEL_CACHE.warmup()
    
    print("="*60)
    print("✅ Server ready!")
    print("▶ Running at: http://127.0.0.1:5000")
    print("="*60)
    
    # Run app với cấu hình tối ưu
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,  # Đặt False để tăng tốc production
        threaded=True
    )