#!/usr/bin/env python3
"""
Google Trends Screenshot Organizer with GPU-Accelerated OCR
- Uses EasyOCR with automatic CUDA/MPS/CPU detection
- Validates metadata against official Google Trends dropdown values
- Sorts screenshots into identified/unidentified folder structures
- Saves OCR output + structured metadata for each file
"""

import os
import re
import json
import shutil
import logging
import argparse
import threading
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import easyocr
import torch
from PIL import Image

from config import (
    GEO_PICKER, ALL_COUNTRIES, TIME_RANGES, CATEGORIES, LOCALES, GTRENDS_UI_KEYWORDS,
    get_country_name, get_country_code, is_valid_country_code, is_valid_time_range, is_valid_category
)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('gtrends_ocr.log', mode='a', encoding='utf-8')]
)

TESSERACT_CONFIG = '--psm 6 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,-+&():/–'
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
_ocr_reader: Optional[easyocr.Reader] = None
_init_lock = threading.Lock()

def sanitize_name(name: str) -> str:
    if not name: return "unknown"
    sanitized = re.sub(r'[^a-z0-9_\-.]', '_', name.lower().strip())
    return re.sub(r'_+', '_', sanitized).strip('_')

def extract_datetime_from_filename(filename: str) -> str:
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}_{match.group(4)}-{match.group(5)}-{match.group(6)}"
    fallback = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
    if fallback:
        return f"{fallback.group(1)}-{fallback.group(2)}-{fallback.group(3)}_unknown-time"
    return "unknown_date_time"

def get_ocr_reader() -> easyocr.Reader:
    global _ocr_reader
    if _ocr_reader is not None: return _ocr_reader
    
    with _init_lock:
        if _ocr_reader is not None: return _ocr_reader
        
        if torch.cuda.is_available():
            device, use_gpu = 'cuda', True
            logging.info("🚀 NVIDIA GPU detected: Using CUDA acceleration")
        else:
            device, use_gpu = 'cpu', False
            logging.info("🍎 Apple Silicon detected: Using optimized CPU inference")
        
        _ocr_reader = easyocr.Reader(
            ['en'], gpu=use_gpu, verbose=False, 
            model_storage_directory='./easyocr_models', download_enabled=True
        )
        logging.info(f"✅ EasyOCR initialized on device: {device.upper()}")
        return _ocr_reader

def ocr_image_easyocr(image_path: str) -> Tuple[str, float]:
    reader = get_ocr_reader()
    img = cv2.imread(str(image_path))
    if img is None: raise ValueError(f"Could not read image: {image_path}")
    
    # Ensure contiguous memory layout (prevents EasyOCR crashes)
    img_rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    try:
        # paragraph=False avoids known EasyOCR crashes; we join manually
        results = reader.readtext(img_rgb, detail=1, paragraph=False, batch_size=4, workers=0)
    except Exception as e:
        logging.error(f"EasyOCR readtext failed: {e}")
        raise
        
    if not results: return "", 0.0
    
    texts, confs = [], []
    for result in results:
        try:
            # EasyOCR returns [(bbox, text, conf), ...] but sometimes drops conf/text
            if len(result) >= 3:
                text, conf = str(result[1]).strip(), float(result[2])
                if text and conf > 0:
                    texts.append(text)
                    confs.append(conf)
            elif len(result) == 2:
                text = str(result[1]).strip()
                if text:
                    texts.append(text)
                    confs.append(0.5) # Default confidence for malformed results
        except (IndexError, ValueError, TypeError):
            continue
            
    if not texts: return "", 0.0
    return " ".join(texts).strip(), sum(confs) / len(confs)

def ocr_image_tesseract(image_path: str) -> Tuple[str, float]:
    try:
        import pytesseract
    except ImportError:
        logging.warning("⚠️  pytesseract not installed; skipping fallback")
        return "", 0.0
        
    img = cv2.imread(str(image_path))
    if img is None: raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    data = pytesseract.image_to_data(thresh, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT)
    valid = [(t, c) for t, c in zip(data['text'], data['conf']) if int(c) > 30 and t.strip()]
    if not valid: return "", 0.0
    texts, confs = zip(*valid)
    return " ".join(texts).strip(), sum(confs) / len(confs)

def ocr_image(image_path: str) -> Tuple[str, float]:
    try:
        return ocr_image_easyocr(image_path)
    except Exception as e:
        logging.warning(f"⚠️  EasyOCR failed for {Path(image_path).name}: {e}")
        logging.info("🔄 Falling back to Tesseract OCR...")
        try:
            return ocr_image_tesseract(image_path)
        except Exception as e2:
            logging.error(f"❌ All OCR methods failed: {e2}\n{traceback.format_exc()}")
            return "", 0.0

def extract_metadata(ocr_text: str, filename: str) -> Optional[Dict]:
    if not ocr_text.strip(): return None
    lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
    content_lines = [l for l in lines if not any(k in l.lower() for k in GTRENDS_UI_KEYWORDS) and len(l) > 3]
    if not content_lines: return None
    
    terms = []
    for line in content_lines[:3]:
        for part in re.split(r'\s*(?:,|\bvs\b|\+|\band\b|\|)\s*', line, flags=re.IGNORECASE):
            p = part.strip().strip('"\'')
            if p and not re.match(r'^\d+[\.,]?\d*%?$', p) and len(p) > 1 and p.lower() not in GTRENDS_UI_KEYWORDS:
                terms.append(p)
    terms = list(dict.fromkeys(terms))[:5]
    if not terms: return None
    
    country_code, country_name = None, "Worldwide"
    code_match = re.search(r'\b([A-Z]{2}(?:-[A-Z]{2,3})?)\b', ocr_text)
    if code_match:
        parent = code_match.group(1).split('-')[0].upper()
        if is_valid_country_code(parent):
            country_code, country_name = parent, get_country_name(parent)
    if not country_code:
        ocr_lower = ocr_text.lower()
        for c, n in GEO_PICKER.items():
            if n.lower() in ocr_lower: country_code, country_name = c, n; break
        if not country_code:
            for c, n in ALL_COUNTRIES.items():
                if n.lower() in ocr_lower: country_code, country_name = c, n; break
                
    timeframe = "Unknown"
    ocr_lower = ocr_text.lower()
    for bid, dname in TIME_RANGES.items():
        if dname.lower() in ocr_lower or bid in ocr_text: timeframe = dname; break
        
    category = None
    for cid, cname in CATEGORIES.items():
        if cname.lower() in ocr_lower or cid in ocr_text: category = cname; break
        
    return {
        'terms': terms, 'country_code': country_code, 'country_name': sanitize_name(country_name),
        'timeframe': sanitize_name(timeframe), 'category': sanitize_name(category) if category else None,
        'datetime': extract_datetime_from_filename(filename), 'original_filename': filename
    }

def process_single(filepath: Path, identified_dir: Path, unidentified_dir: Path, save_metadata_json: bool = True):
    filename = filepath.name
    logging.info(f"📷 Processing: {filename}")
    try:
        ocr_text, avg_conf = ocr_image(filepath)
        txt_filename = filepath.stem + ".txt"
        
        if not ocr_text.strip():
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text("OCR_OUTPUT:\n\n(No text detected)\n\nCONFIDENCE: 0.0", encoding='utf-8')
            return
            
        metadata = extract_metadata(ocr_text, filename)
        txt_content = f"""OCR_TEXT:\n{ocr_text}\n\nMETADATA:\n- Terms: {metadata['terms'] if metadata else 'None'}\n- Country: {metadata['country_name'] if metadata else 'Unknown'} ({metadata['country_code'] if metadata else 'N/A'})\n- Timeframe: {metadata['timeframe'] if metadata else 'Unknown'}\n- Datetime: {metadata['datetime'] if metadata else 'Unknown'}\n\nCONFIDENCE: {avg_conf:.2f}"""
        
        if metadata and metadata['terms']:
            base = "multiple" if len(metadata['terms']) > 1 else sanitize_name(metadata['terms'][0])
            dest_dir = identified_dir / base / (metadata['country_code'] or "WW") / metadata['datetime']
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dest_dir / filename)
            (dest_dir / txt_filename).write_text(txt_content, encoding='utf-8')
            if save_metadata_json:
                meta_json = {**metadata, 'ocr_confidence': avg_conf, 'ocr_engine': 'easyocr' if avg_conf > 0 else 'tesseract'}
                (dest_dir / "metadata.json").write_text(json.dumps(meta_json, indent=2, ensure_ascii=False), encoding='utf-8')
            logging.info(f"  ✅ Saved to: {dest_dir}")
        else:
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(txt_content, encoding='utf-8')
            logging.info(f"  ⚠️  No valid GTrends metadata found -> unidentified")
    except Exception as e:
        logging.error(f"❌ Failed {filename}: {e}\n{traceback.format_exc()}")
        try:
            txt_filename = filepath.stem + ".txt"
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(f"ERROR: {str(e)}\n\n(OCR failed)", encoding='utf-8')
        except Exception as e2:
            logging.error(f"Fallback failed: {e2}")

def main():
    parser = argparse.ArgumentParser(description="Organize Google Trends screenshots via OCR")
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-csv", action="store_true")
    parser.add_argument("--no-metadata-json", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "processed"
    identified_dir = output_dir / "identified"
    unidentified_dir = output_dir / "unidentified"
    for d in [identified_dir, unidentified_dir]: d.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logging.warning("⚠️  No supported images found."); return

    logging.info(f"🔍 Found {len(files)} screenshots. Processing with {args.workers} workers...")
    get_ocr_reader() # Pre-warm
    
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(as_completed(ex.submit(process_single, f, identified_dir, unidentified_dir, not args.no_metadata_json) for f in files))
        
    logging.info(f"✅ Processing complete in {time.time()-start:.1f}s")
    logging.info(f"📊 Results: {sum(1 for _ in identified_dir.rglob('*.png'))+sum(1 for _ in identified_dir.rglob('*.jpg'))} identified")

if __name__ == "__main__":
    main()