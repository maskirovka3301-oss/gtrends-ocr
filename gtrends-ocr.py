#!/usr/bin/env python3
"""
Google Trends Screenshot Organizer using HuggingFace Kimi-K2.6 Vision Model
- Uses MoonshotAI's Kimi-K2.6 via HuggingFace Inference API
- Extracts text from mobile screenshots with high accuracy
- Validates metadata against official Google Trends dropdown values
"""

import os
import re
import json
import shutil
import logging
import argparse
import base64
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from openai import OpenAI
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

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
MODEL_NAME = "moonshotai/Kimi-K2.6"
OCR_PROMPT = "Please extract all text from the image and output them as text, comma separated"

# HuggingFace API client
_hf_client: Optional[OpenAI] = None

def get_hf_client() -> OpenAI:
    """Initialize HuggingFace OpenAI-compatible client"""
    global _hf_client
    if _hf_client is not None:
        return _hf_client
    
    api_key = os.getenv('HUGGINGFACE_API_KEY')
    if not api_key:
        raise ValueError(
            "HUGGINGFACE_API_KEY environment variable not set!\n"
            "Get a token at: https://huggingface.co/settings/tokens\n"
            "Then run: export HUGGINGFACE_API_KEY='hf_xxxxx'"
        )
    
    _hf_client = OpenAI(
        api_key=api_key,
        base_url="https://api-inference.huggingface.co/v1"
    )
    logging.info("✅ HuggingFace client initialized")
    return _hf_client

def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def ocr_with_kimi(image_path: str) -> Tuple[str, float]:
    """
    Extract text using Kimi-K2.6 vision model via HuggingFace API
    Returns: (extracted_text, confidence_score)
    """
    client = get_hf_client()
    
    # Convert image to base64
    try:
        image_base64 = image_to_base64(image_path)
    except Exception as e:
        logging.error(f"Failed to encode image: {e}")
        return "", 0.0
    
    # Prepare message with image
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': OCR_PROMPT},
                {
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'},
                },
            ],
        }
    ]
    
    try:
        logging.debug(f"📤 Sending image to Kimi-K2.6...")
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=False,
            max_tokens=4096,
            extra_body={'thinking': {'type': 'disabled'}}  # Instant mode for speed
        )
        
        extracted_text = response.choices[0].message.content.strip()
        
        # Kimi doesn't provide confidence, estimate based on text length
        confidence = min(0.95, 0.5 + (len(extracted_text) / 1000)) if extracted_text else 0.0
        
        logging.debug(f"📥 Extracted {len(extracted_text)} chars")
        return extracted_text, confidence
        
    except Exception as e:
        logging.error(f"Kimi-K2.6 API error: {e}")
        logging.debug(f"Full error: {traceback.format_exc()}")
        return "", 0.0

def sanitize_name(name: str) -> str:
    """Make folder/file names filesystem-safe"""
    if not name: return "unknown"
    sanitized = re.sub(r'[^a-z0-9_\-.]', '_', name.lower().strip())
    return re.sub(r'_+', '_', sanitized).strip('_')

def extract_datetime_from_filename(filename: str) -> str:
    """Extract datetime from screenshot filename"""
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}_{match.group(4)}-{match.group(5)}-{match.group(6)}"
    fallback = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
    if fallback:
        return f"{fallback.group(1)}-{fallback.group(2)}-{fallback.group(3)}_unknown-time"
    return "unknown_date_time"

def extract_metadata(ocr_text: str, filename: str) -> Optional[Dict]:
    """Parse Google Trends metadata from OCR output"""
    if not ocr_text.strip(): 
        return None
    
    # Log preview for debugging
    logging.info(f"📝 Extracted text preview: '{ocr_text[:200]}...'")
    
    # Split by comma (as requested in prompt) and clean
    raw_items = [item.strip() for item in ocr_text.split(',') if item.strip()]
    
    # Filter out UI keywords and noise
    content_items = [
        item for item in raw_items 
        if len(item) > 2 
        and item.lower() not in GTRENDS_UI_KEYWORDS
        and not re.match(r'^\d+[\.,]?\d*%?$', item)
    ]
    
    if not content_items:
        return None
    
    # Extract search terms (usually first few meaningful items)
    terms = []
    for item in content_items[:5]:
        # Skip common non-term patterns
        if (item.lower() in ['interest over time', 'interest by region', 'related topics', 
                             'related queries', 'web search', 'top', 'rising'] or
            re.match(r'^(mon|tue|wed|thu|fri|sat|sun|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+\d+', item.lower())):
            continue
        terms.append(item)
    
    terms = list(dict.fromkeys(terms))[:3]  # Dedupe, max 3 terms
    if not terms:
        return None
    
    # Extract country
    country_code, country_name = None, "Worldwide"
    ocr_lower = ocr_text.lower()
    
    # Check for country codes
    code_match = re.search(r'\b([A-Z]{2}(?:-[A-Z]{2,3})?)\b', ocr_text)
    if code_match:
        parent = code_match.group(1).split('-')[0].upper()
        if is_valid_country_code(parent):
            country_code, country_name = parent, get_country_name(parent)
    
    # Check for country names
    if not country_code:
        for c, n in GEO_PICKER.items():
            if n.lower() in ocr_lower:
                country_code, country_name = c, n
                break
        if not country_code:
            for c, n in ALL_COUNTRIES.items():
                if n.lower() in ocr_lower:
                    country_code, country_name = c, n
                    break
    
    # Extract timeframe
    timeframe = "Unknown"
    for bid, dname in TIME_RANGES.items():
        if dname.lower() in ocr_lower or bid in ocr_text:
            timeframe = dname
            break
    
    # Extract category
    category = None
    for cid, cname in CATEGORIES.items():
        if cname.lower() in ocr_lower or cid in ocr_text:
            category = cname
            break
    
    return {
        'terms': terms, 
        'country_code': country_code, 
        'country_name': sanitize_name(country_name),
        'timeframe': sanitize_name(timeframe), 
        'category': sanitize_name(category) if category else None,
        'datetime': extract_datetime_from_filename(filename), 
        'original_filename': filename
    }

def process_single(filepath: Path, identified_dir: Path, unidentified_dir: Path, save_metadata_json: bool = True):
    """Process a single screenshot"""
    filename = filepath.name
    logging.info(f"📷 Processing: {filename}")
    
    try:
        # Run OCR with Kimi-K2.6
        ocr_text, avg_conf = ocr_with_kimi(filepath)
        txt_filename = filepath.stem + ".txt"
        
        if not ocr_text.strip():
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(
                "OCR_OUTPUT:\n\n(No text detected)\n\nCONFIDENCE: 0.0", 
                encoding='utf-8'
            )
            return
        
        # Extract metadata
        metadata = extract_metadata(ocr_text, filename)
        
        # Prepare output content
        txt_content = f"""OCR_TEXT:\n{ocr_text}\n\nMETADATA:\n- Terms: {metadata['terms'] if metadata else 'None'}\n- Country: {metadata['country_name'] if metadata else 'Unknown'} ({metadata['country_code'] if metadata else 'N/A'})\n- Timeframe: {metadata['timeframe'] if metadata else 'Unknown'}\n- Datetime: {metadata['datetime'] if metadata else 'Unknown'}\n\nCONFIDENCE: {avg_conf:.2f}"""
        
        if metadata and metadata['terms']:
            # Determine folder structure
            base = "multiple" if len(metadata['terms']) > 1 else sanitize_name(metadata['terms'][0])
            dest_dir = identified_dir / base / (metadata['country_code'] or "WW") / metadata['datetime']
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Save files
            shutil.copy2(filepath, dest_dir / filename)
            (dest_dir / txt_filename).write_text(txt_content, encoding='utf-8')
            
            if save_metadata_json:
                meta_json = {**metadata, 'ocr_confidence': avg_conf, 'ocr_engine': 'kimi-k2.6'}
                (dest_dir / "metadata.json").write_text(
                    json.dumps(meta_json, indent=2, ensure_ascii=False), 
                    encoding='utf-8'
                )
            
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
            (unidentified_dir / txt_filename).write_text(
                f"ERROR: {str(e)}\n\n(OCR failed)", 
                encoding='utf-8'
            )
        except Exception as e2:
            logging.error(f"Fallback failed: {e2}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Organize Google Trends screenshots using Kimi-K2.6 OCR")
    parser.add_argument("input_dir", type=str, help="Path to folder containing screenshots")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder (default: input_dir/processed)")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    parser.add_argument("--no-metadata-json", action="store_true", help="Skip metadata.json files")
    args = parser.parse_args()

    # Verify API key
    if not os.getenv('HUGGINGFACE_API_KEY'):
        logging.error("❌ HUGGINGFACE_API_KEY not set!")
        logging.error("Get a token at: https://huggingface.co/settings/tokens")
        logging.error("Then run: export HUGGINGFACE_API_KEY='hf_xxxxx'")
        return

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "processed"
    identified_dir = output_dir / "identified"
    unidentified_dir = output_dir / "unidentified"
    
    for d in [identified_dir, unidentified_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Find image files
    files = [
        f for f in input_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    if not files:
        logging.warning("⚠️  No supported images found.")
        return

    logging.info(f"🔍 Found {len(files)} screenshots. Processing with {args.workers} workers...")
    
    # Initialize client
    try:
        get_hf_client()
    except ValueError as e:
        logging.error(str(e))
        return
    
    # Process files
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_single, f, identified_dir, unidentified_dir, not args.no_metadata_json) 
            for f in files
        ]
        for i, future in enumerate(as_completed(futures), 1):
            if i % 5 == 0:
                logging.info(f"⏳ Progress: {i}/{len(files)}")
        
    elapsed = time.time() - start
    identified_count = sum(1 for _ in identified_dir.rglob('*.png')) + sum(1 for _ in identified_dir.rglob('*.jpg'))
    
    logging.info(f"✅ Processing complete in {elapsed:.1f}s")
    logging.info(f"📊 Results: {identified_count} identified, {len(files)-identified_count} unidentified")

if __name__ == "__main__":
    main()