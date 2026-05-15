#!/usr/bin/env python3
"""
Google Trends Screenshot Organizer using Qwen/Qwen3-VL-2B-Instruct
- Robust 2B VLM with strong instruction-following & JSON adherence
- Downloads model ONCE to ./models/ directory
- Auto-retries with contextual hints if GTrends fields are missing
- Injects filename-extracted datetime into the final JSON
- Extracts chart time series via deterministic pixel scanning
- NEW: --compile switch to merge all valid timeseries into a nested JSON structure
"""
import os
import re
import json
import shutil
import logging
import argparse
import time
import traceback
import sys
import platform
import threading
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

# Dependency guards (includes chart extractor dependencies)
for pkg in ["transformers", "torch", "accelerate", "pydantic", "tiktoken", "regex", 
            "packaging", "huggingface_hub", "numpy", "cv2", "scipy"]:
    try:
        __import__(pkg)
    except ImportError:
        logging.error(f"❌ {pkg} is required")
        logging.error(f"Run: pip install {pkg.replace('_', '-')}")
        raise

import torch
import numpy as np
import cv2
from transformers import AutoProcessor, AutoModelForImageTextToText
from huggingface_hub import snapshot_download
from config import (
    GEO_PICKER, ALL_COUNTRIES, TIME_RANGES, CATEGORIES, LOCALES, GTRENDS_UI_KEYWORDS,
    get_country_name, get_country_code, is_valid_country_code, is_valid_time_range, is_valid_category
)
from chart_extract import extract_and_interpolate_trends

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('gtrends_ocr.log', mode='a', encoding='utf-8')]
)

SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp', '.gif'}
MODEL_REPO = "Qwen/Qwen3-VL-2B-Instruct"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_MODEL_DIR = os.path.join(SCRIPT_DIR, "models", "Qwen3-VL-2B-Instruct")

_model = None
_processor = None
_model_loaded = False
_load_lock = threading.Lock()

def ensure_model_downloaded():
    """Download model once to local directory. Skip if already present."""
    config_path = os.path.join(LOCAL_MODEL_DIR, "config.json")
    if os.path.exists(config_path):
        logging.info(f"💾 Using cached model from: {LOCAL_MODEL_DIR}")
        os.environ["HF_HUB_OFFLINE"] = "1"
        return
    logging.info(f"📥 Model not found locally. Downloading to {LOCAL_MODEL_DIR}...")
    logging.info("⚠️  This will download ~4.5GB. It only happens once.")
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=LOCAL_MODEL_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.pt", "*.bin", "*.msgpack", "*.gguf", "*.h5"]
    )
    logging.info("✅ Model download complete. Future runs will load instantly from disk.")
    os.environ["HF_HUB_OFFLINE"] = "1"

def load_model():
    """Load model with proper synchronization. Returns (model, processor)"""
    global _model, _processor, _model_loaded
    if _model_loaded and _model is not None and _processor is not None:
        return _model, _processor

    with _load_lock:
        if _model_loaded and _model is not None and _processor is not None:
            return _model, _processor

        ensure_model_downloaded()
        logging.info("🔄 Loading Qwen3-VL-2B-Instruct...")
        try:
            is_mac = sys.platform == "darwin" and platform.machine() == "arm64"
            torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16
            device_map = "mps" if is_mac else "auto"

            logging.info(f"{'🍎 Apple Silicon' if is_mac else '💻 GPU/CPU'} detected: Using {device_map.upper()} with {torch_dtype}")

            _processor = AutoProcessor.from_pretrained(
                LOCAL_MODEL_DIR,
                trust_remote_code=True,
                local_files_only=True
            )
            _model = AutoModelForImageTextToText.from_pretrained(
                LOCAL_MODEL_DIR,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                local_files_only=True
            )
            _model.eval()
            _model_loaded = True
            logging.info("✅ Model loaded successfully!")
            return _model, _processor
        except Exception as e:
            logging.error(f"❌ Failed to load model: {e}")
            logging.error(traceback.format_exc())
            raise

def extract_strict_json(text: str) -> Optional[Dict]:
    """Extract JSON strictly according to the required format"""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        json_match = re.search(r'\{[^{}]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        return None

def sanitize_name(name: str) -> str:
    if not name: return "unknown"
    return re.sub(r'_+', '_', re.sub(r'[^a-z0-9_\-.]', '_', str(name).lower().strip())).strip('_')

def extract_datetime_from_filename(filename: str) -> str:
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})', filename)
    if match: return f"{match.group(1)}-{match.group(2)}-{match.group(3)}_{match.group(4)}-{match.group(5)}-{match.group(6)}"
    fallback = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
    if fallback: return f"{fallback.group(1)}-{fallback.group(2)}-{fallback.group(3)}_unknown-time"
    return "unknown_date_time"

def extract_metadata_with_qwen3vl(image_path: str, filename: str) -> Tuple[Optional[Dict], float]:
    """Process image with Qwen3-VL-2B. Auto-retries with hints if GTrends fields are missing."""
    model, processor = load_model()
    try:
        image = Image.open(image_path).convert("RGB")

        system_prompt = "You are a precise data extraction assistant. Output ONLY valid JSON with no additional text, explanations, or markdown formatting."
        user_prompt = (
            "Analyze this Google Trends screenshot and extract EXACTLY these fields in JSON format:\n"
            "1. is_google_trends: true ONLY if you see \"Google Trends\" logo/text OR \"Interest over time\" chart. Otherwise false.\n"
            "2. search_terms: Array of strings. Extract ALL search terms being compared (top of page). If partially visible, find complete terms elsewhere on page. Max 5 terms.\n"
            "3. country: Selected country/region from dropdown (e.g., \"Worldwide\", \"United States\", \"Brazil\"). null if not visible.\n"
            "4. date_range: Selected timeframe from dropdown (e.g., \"Past day\", \"Past 30 days\"). null if not visible.\n"
            "RULES:\n"
            "- Output ONLY the JSON object with these 4 exact keys\n"
            "- search_terms MUST be an array, even for single term: [\"term\"]\n"
            "- Use null for missing values, NOT empty string or \"N/A\"\n"
            "- NEVER include ANY text before or after the JSON\n"
            "- NEVER use markdown formatting\n"
            "- If unsure about Google Trends identification, set is_google_trends to false\n"
            "EXAMPLE OUTPUT (valid):\n"
            "{\"is_google_trends\": true, \"search_terms\": [\"term1\", \"term2\"], \"country\": \"United States\", \"date_range\": \"Past 30 days\"}\n"
            "EXAMPLE OUTPUT (not Google Trends):\n"
            "{\"is_google_trends\": false, \"search_terms\": [], \"country\": null, \"date_range\": null}\n"
            "Now analyze the screenshot and return ONLY the JSON object:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_prompt}]}
        ]

        prompt_text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=prompt_text, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False, temperature=0.0)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        generated_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        logging.info(f"🔍 Raw model output: {repr(generated_text[:300])}")

        current_json = extract_strict_json(generated_text)
        if current_json is None:
            logging.warning("⚠️  Model did not return valid JSON initially.")
            return None, 0.0

        required_keys = ["is_google_trends", "search_terms", "country", "date_range"]
        if not all(key in current_json for key in required_keys):
            logging.warning("⚠️  JSON is missing required keys initially.")
            return None, 0.0

        if not isinstance(current_json["search_terms"], list):
            current_json["search_terms"] = []

        # Retry logic for missing fields if identified as GTrends
        if current_json.get("is_google_trends"):
            max_retries = 2
            for attempt in range(max_retries):
                missing_fields = []
                terms = current_json.get("search_terms")
                if not terms or (isinstance(terms, list) and len(terms) == 0):
                    missing_fields.append("search_terms")
                if not current_json.get("country") or str(current_json.get("country")).lower() in ["null", "none", ""]:
                    missing_fields.append("country")
                if not current_json.get("date_range") or str(current_json.get("date_range")).lower() in ["null", "none", ""]:
                    missing_fields.append("date_range")

                if not missing_fields:
                    break

                logging.info(f"🔁 Retry {attempt+1}/{max_retries}: Missing {missing_fields}. Re-prompting with hints...")
                hints = []
                if "search_terms" in missing_fields:
                    hints.append("- search_terms: Look at the top search bar or comparison chips. Extract exact query text.")
                if "country" in missing_fields:
                    hints.append("- country: Look at the filter bar dropdown. Usually says 'Worldwide', 'United States', etc.")
                if "date_range" in missing_fields:
                    hints.append("- date_range: Look next to the country filter. Usually says 'Past 30 days', 'Past 12 months', etc.")

                retry_prompt = (
                    f"You previously analyzed this screenshot and returned:\n{json.dumps(current_json)}\n"
                    f"However, these fields are missing or empty: {', '.join(missing_fields)}.\n"
                    f"Please examine the image again carefully:\n" + "\n".join(hints) + "\n"
                    f"Return ONLY a JSON object with the missing fields filled in. I will merge it with your previous output. Do not include markdown or extra text."
                )

                retry_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": retry_prompt}]}
                ]
                retry_text = processor.apply_chat_template(retry_messages, add_generation_prompt=True, tokenize=False)
                retry_inputs = processor(text=retry_text, images=image, return_tensors="pt").to(model.device)

                with torch.no_grad():
                    retry_ids = model.generate(**retry_inputs, max_new_tokens=256, do_sample=False, temperature=0.0)
                retry_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(retry_inputs.input_ids, retry_ids)]
                retry_output = processor.batch_decode(retry_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
                logging.info(f"🔍 Retry output: {repr(retry_output[:200])}")

                new_json = extract_strict_json(retry_output)
                if new_json and isinstance(new_json, dict):
                    current_json.update(new_json)
                    if not isinstance(current_json.get("search_terms"), list):
                        current_json["search_terms"] = []
                else:
                    logging.warning("⚠️  Retry did not return valid JSON. Stopping retries.")
                    break

        current_json["filename_datetime"] = extract_datetime_from_filename(filename)
        confidence = 0.85 if current_json.get('is_google_trends', False) else 0.4
        return current_json, confidence

    except Exception as e:
        logging.error(f"Inference error: {e}")
        logging.debug(traceback.format_exc())
        return None, 0.0

def validate_and_normalize_metadata(extracted: Dict, filename: str) -> Optional[Dict]:
    """Validate and normalize extracted metadata against Google Trends constants"""
    if not extracted.get('is_google_trends'):
        logging.debug("❌ Not identified as Google Trends screenshot")
        return None

    terms = extracted.get('search_terms', [])
    if not isinstance(terms, list): terms = []
    terms = [str(t).strip() for t in terms if t and str(t).strip()]
    if not terms:
        logging.debug("⚠️  No search terms found")
        return None
    terms = list(dict.fromkeys(terms))[:5]

    country_code = None
    country_name = "Worldwide"
    raw_country = extracted.get('country', '')
    if raw_country:
        country_str = str(raw_country).lower().strip()
        code_match = re.match(r'^([A-Z]{2})(?:-[A-Z]{2,3})?$', str(raw_country).upper())
        if code_match:
            parent_code = code_match.group(1)
            if is_valid_country_code(parent_code):
                country_code = parent_code
                country_name = get_country_name(parent_code)
        if not country_code:
            for c, n in GEO_PICKER.items():
                if n.lower() == country_str or country_str in n.lower():
                    country_code = c
                    country_name = n
                    break
            if not country_code:
                for c, n in ALL_COUNTRIES.items():
                    if n.lower() == country_str or country_str in n.lower():
                        country_code = c
                        country_name = n
                        break

    timeframe = "Unknown"
    raw_timeframe = extracted.get('date_range', '')
    if raw_timeframe:
        timeframe_str = str(raw_timeframe).lower().strip()
        for backend_id, display_name in TIME_RANGES.items():
            if display_name.lower() == timeframe_str or timeframe_str in display_name.lower():
                timeframe = display_name
                break
            if backend_id.lower() == timeframe_str:
                timeframe = display_name
                break

    return {
        'terms': terms,
        'country_code': country_code,
        'country_name': sanitize_name(country_name),
        'timeframe': sanitize_name(timeframe),
        'datetime': extracted.get('filename_datetime', extract_datetime_from_filename(filename)),
        'original_filename': filename,
        'raw_country': str(raw_country) if raw_country else None,
        'raw_timeframe': str(raw_timeframe) if raw_timeframe else None
    }

def compile_metadata_json(search_dir: str, output_path: str):
    """Recursively scans a directory for metadata.json, filters valid timeseries, and merges into a nested JSON."""
    identified_dir = Path(search_dir)
    if not identified_dir.exists():
        logging.error(f"❌ Directory '{identified_dir}' not found.")
        return

    output_file = Path(output_path)
    compiled_data = {}
    processed_count = 0
    skipped_count = 0

    logging.info(f"🔍 Scanning '{identified_dir}' for metadata.json files...")
    for json_path in sorted(identified_dir.rglob("metadata.json")):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            series = data.get('chart_time_series')
            if not isinstance(series, list) or len(series) == 0:
                skipped_count += 1
                continue

            terms = data.get('terms', [])
            country_code = data.get('country_code') or "WW"
            dt = data.get('datetime', "unknown_datetime")

            # Build the exact minimal payload structure requested
            minimal_payload = {
                "chart_time_series": series,
                "country_name": data.get("country_name", "unknown"),
                "original_filename": data.get("original_filename", ""),
                "timeframe": data.get("timeframe", "unknown")
            }

            for term in terms:
                term_key = str(term).strip()
                if not term_key:
                    continue
                    
                compiled_data.setdefault(term_key, {}).setdefault(country_code, {})[dt] = minimal_payload

            processed_count += 1
        except Exception as e:
            logging.warning(f"⚠️ Failed to process {json_path}: {e}")

    logging.info(f"✅ Compiled {processed_count} entries (skipped {skipped_count} without valid timeseries)")
    logging.info(f"💾 Writing nested structure to {output_file}...")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(compiled_data, f, indent=2, ensure_ascii=False, sort_keys=True)

    logging.info(f" Compilation complete: hierarchical JSON saved to {output_file}")

def process_single(filepath: Path, identified_dir: Path, unidentified_dir: Path, save_metadata_json: bool = True):
    filename = filepath.name
    logging.info(f"📷 Processing: {filename}")
    try:
        extracted, confidence = extract_metadata_with_qwen3vl(filepath, filename)
        txt_filename = filepath.stem + ".txt"

        if extracted is None:
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(
                "OCR_OUTPUT:\n(No structured data extracted)\nCONFIDENCE: 0.0",
                encoding='utf-8'
            )
            return

        logging.info(f"📝 Raw extraction: {json.dumps(extracted, indent=2)}")
        metadata = validate_and_normalize_metadata(extracted, filename)

        if metadata and metadata['terms']:
            # 📈 Extract deterministic chart timeseries using pixel analysis
            chart_series = None
            try:
                logging.info("📈 Extracting chart timeseries via pixel scanning...")
                chart_series = extract_and_interpolate_trends(str(filepath))
                if chart_series and len(chart_series) == 2048:
                    logging.info("✅ Chart extraction successful: 2048 points")
                else:
                    logging.warning("⚠️ Chart extraction returned unexpected data length")
                    chart_series = None
            except Exception as e:
                logging.warning(f"⚠️ Chart extraction failed: {e}")
                chart_series = None

            base = "multiple" if len(metadata['terms']) > 1 else sanitize_name(metadata['terms'][0])
            dest_dir = identified_dir / base / (metadata['country_code'] or "WW") / metadata['datetime']
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(filepath, dest_dir / filename)

            series_summary = f"{len(chart_series)} points" if chart_series else "Failed/None"
            txt_content = f"""RAW_EXTRACTION:\n{json.dumps({k: v for k, v in extracted.items()}, indent=2)}\nCHART_TIME_SERIES: {series_summary}\nNORMALIZED_METADATA:\n- Terms: {metadata['terms']}\n- Country: {metadata['country_name']} ({metadata['country_code'] or 'N/A'})\n- Timeframe: {metadata['timeframe']}\n- Datetime: {metadata['datetime']}\nCONFIDENCE: {confidence:.2f}\nOCR_ENGINE: qwen3-vl-2b\n"""
            (dest_dir / txt_filename).write_text(txt_content, encoding='utf-8')

            if save_metadata_json:
                meta_json = {
                    **metadata,
                    'ocr_confidence': confidence,
                    'ocr_engine': 'qwen3-vl-2b',
                    'is_google_trends': True,
                    'chart_time_series': chart_series
                }
                (dest_dir / "metadata.json").write_text(
                    json.dumps(meta_json, indent=2, ensure_ascii=False),
                    encoding='utf-8'
                )
            logging.info(f"  ✅ Saved to: {dest_dir}")
        else:
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(
                f"RAW_EXTRACTION:\n{json.dumps(extracted, indent=2)}\nSTATUS: Invalid or missing metadata",
                encoding='utf-8'
            )
            logging.info(f"  ⚠️  No valid GTrends metadata -> unidentified")
    except Exception as e:
        logging.error(f"❌ Failed {filename}: {e}\n{traceback.format_exc()}")
        try:
            txt_filename = filepath.stem + ".txt"
            shutil.copy2(filepath, unidentified_dir / filename)
            (unidentified_dir / txt_filename).write_text(
                f"ERROR: {str(e)}\n(Processing failed)",
                encoding='utf-8'
            )
        except Exception as e2:
            logging.error(f"Fallback failed: {e2}")

def main():
    parser = argparse.ArgumentParser(description="Organize Google Trends screenshots using Qwen3-VL-2B")
    parser.add_argument("input_dir", nargs='?', type=str, help="Path to folder containing screenshots")
    parser.add_argument("--output_dir", type=str, default=None, help="Output folder")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (default: 1)")
    parser.add_argument("--no-metadata-json", action="store_true", help="Skip metadata.json files")
    parser.add_argument("--compile", nargs=2, metavar=("SEARCH_DIR", "OUTPUT_JSON"),
                        help="Recursively compile all valid timeseries from a directory into a nested JSON file.")
    args = parser.parse_args()

    # Handle --compile as a standalone post-processing step (no model loading required)
    if args.compile:
        compile_metadata_json(args.compile[0], args.compile[1])
        return

    if not args.input_dir:
        parser.error("input_dir is required unless using --compile")
        return

    logging.info("🔄 Loading model before processing screenshots...")
    load_model()
    logging.info("✅ Model is fully loaded. Starting screenshot processing...")

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "processed"
    identified_dir = output_dir / "identified"
    unidentified_dir = output_dir / "unidentified"
    for d in [identified_dir, unidentified_dir]:
        d.mkdir(parents=True, exist_ok=True)

    files = [f for f in input_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS]
    if not files:
        logging.warning("⚠️  No supported images found.")
        return

    logging.info(f"🔍 Found {len(files)} screenshots. Processing with {args.workers} worker(s)...")
    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_single, f, identified_dir, unidentified_dir, not args.no_metadata_json) for f in files]
        for i, future in enumerate(as_completed(futures), 1):
            if i % 5 == 0:
                logging.info(f"⏳ Progress: {i}/{len(files)}")
            future.result()

    elapsed = time.time() - start
    identified_count = sum(1 for _ in identified_dir.rglob('*.txt'))
    logging.info(f"✅ Processing complete in {elapsed:.1f}s")
    logging.info(f"📊 Results: {identified_count} identified, {len(files)-identified_count} unidentified")

if __name__ == "__main__":
    main()