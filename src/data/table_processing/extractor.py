"""
Table data extraction from images using API endpoints and Transformers.

This module handles the extraction of structured table data from images
using remote API endpoints and local Transformers models as fallback.
"""

import logging
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
import platform
import httpx

# Import our config
from ...config import OCR_API_URL, OCR_CLEANUP_URL

logger = logging.getLogger(__name__)

# Temporary flag to disable Transformers (set to False to use API only)
ENABLE_TRANSFORMERS = False

# Transformers model constants
TRANSFORMERS_MODEL_REPO = "nanonets/Nanonets-OCR-s"
SYSTEM_PROMPT = """You are an expert at extracting structured table data from images. Your task is to convert table images into basic HTML format using only <table>, <tr>, and <td> tags.

IMPORTANT REQUIREMENTS:
1. Use ONLY <table>, <tr>, and <td> tags (no thead, tbody, th)
2. Preserve ALL text content exactly as it appears in the image
3. Handle merged cells using colspan and rowspan attributes
4. Do not include explanations or text outside of HTML tags"""

USER_PROMPT = """Please extract all tables from this image and return them in basic HTML format.

Use only <table>, <tr>, and <td> tags with appropriate colspan/rowspan for merged cells.
Preserve all text, numbers, symbols, and formatting exactly as shown.

If there are multiple tables, include all of them in the response.
If there are no tables in the image, return an empty string."""

# Global variables for Transformers model
_transformers_model = None
_transformers_tokenizer = None
_transformers_processor = None


def _load_transformers_model():
    """Load the Transformers model for OCR."""
    global _transformers_model, _transformers_tokenizer, _transformers_processor
    
    if _transformers_model is not None:
        return _transformers_model, _transformers_tokenizer, _transformers_processor

    try:
        logger.info(f"Loading transformers model: {TRANSFORMERS_MODEL_REPO}")
        
        # Use parameters from Hugging Face guide, but handle flash attention conditionally
        model_kwargs = {
            "torch_dtype": "auto", 
            "device_map": "auto"
        }
        
        # Only use flash_attention_2 if available (not on Windows)
        if platform.system().lower() != "win":
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Using flash_attention_2")
            except ImportError:
                logger.info("flash_attn not available, using default attention")
        else:
            logger.info("Windows detected, using default attention")
        
        _transformers_model = AutoModelForImageTextToText.from_pretrained(
            TRANSFORMERS_MODEL_REPO, 
            **model_kwargs
        )
        _transformers_model.eval()
        
        logger.info("Loading tokenizer...")
        _transformers_tokenizer = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL_REPO)
        
        logger.info("Loading processor...")
        _transformers_processor = AutoProcessor.from_pretrained(TRANSFORMERS_MODEL_REPO)
        
        logger.info("✅ Transformers model loaded successfully")
        return _transformers_model, _transformers_tokenizer, _transformers_processor
        
    except Exception as e:
        logger.error(f"Failed to load transformers model: {e}")
        raise


class TableExtractor:
    """
    Handles extraction of structured data from table images using API endpoints and Transformers.
    
    Supports remote API endpoints and local Transformers models for table extraction.
    """
    
    def __init__(self, 
                 api_url: str = None,
                 confidence_threshold: float = 0.8,
                 cleanup_url: str = None):
        """
        Initialize the table extractor.
        
        Args:
            api_url: Remote API endpoint URL (defaults to OCR_API_URL from config)
            confidence_threshold: Minimum confidence for table structure recognition
        """
        self.api_url = api_url or OCR_API_URL
        self.confidence_threshold = confidence_threshold
        self.cleanup_url = cleanup_url or OCR_CLEANUP_URL or None
        
        logger.info(f"Initialized table extractor (api_url={self.api_url}, "
                   f"confidence_threshold={confidence_threshold}, cleanup_url={self.cleanup_url})")
    
    def extract_tables_from_image(self, image_path: str) -> Optional[str]:
        """
        Extract table HTML from an image.
        
        Args:
            image_path: Path to table image file
            
        Returns:
            Raw HTML string containing tables, or None if extraction failed
        """
        try:
            logger.info(f"🔍 Starting table extraction for: {image_path}")
            
            # Primary: Remote API endpoint with retries
            if self.api_url:
                for attempt in range(3):  # 3 retries
                    try:
                        logger.info(f"🔄 Attempting API extraction (attempt {attempt + 1}/3)...")
                        html_from_api = self._ocr_image_via_api(image_path)
                        if html_from_api and html_from_api.strip():
                            logger.info("✅ API extraction successful")
                            return html_from_api
                        else:
                            logger.warning("⚠️ API returned empty result")
                            break  # Don't retry if API returns empty
                    except Exception as e:
                        logger.warning(f"❌ API extraction failed (attempt {attempt + 1}/3): {e}")
                        if attempt == 2:  # Last attempt
                            logger.error("❌ API extraction failed after 3 attempts")

            # Fallback: Transformers (only if enabled)
            if ENABLE_TRANSFORMERS:
                try:
                    logger.info("🔄 Attempting Transformers extraction...")
                    html_content = self._ocr_image_to_html_transformers(image_path)
                    if html_content and html_content.strip():
                        logger.info("✅ Transformers extraction successful")
                        return html_content
                    else:
                        logger.warning("⚠️ Transformers returned empty result")
                except Exception as e:
                    logger.error(f"❌ Transformers extraction failed: {e}")
            else:
                logger.info("🔄 Transformers disabled, skipping fallback")

            logger.error("❌ All extraction methods failed")
            return None

        except Exception as e:
            logger.error(f"Failed to extract tables from image {image_path}: {e}")
            return None

    def extract_tables_via_api(self, image_path: str) -> Optional[dict]:
        """Return the full API JSON if available (includes text + structured tables)."""
        try:
            return self._ocr_call_api(image_path)
        except Exception as e:
            logger.warning(f"API structured call failed: {e}")
            return None

    def _ocr_image_via_api(self, image_path: str) -> Optional[str]:
        """Call API and return HTML text field if present."""
        data = self._ocr_call_api(image_path)
        if not data:
            return None
        text = data.get("text") or data.get("html")
        return text

    def _ocr_call_api(self, image_path: str) -> Optional[dict]:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not self.api_url:
            raise RuntimeError("API URL is not configured")
        
        logger.info(f"🔄 Calling OCR API: {self.api_url}")
        with httpx.Client(timeout=120.0) as client:
            # Prepare form data with required parameters
            files = {"image": (Path(image_path).name, open(image_path, "rb"), "image/png")}
            data = {
                "mode": "result_table",
                "max_new_tokens": "8000",
                "do_sample": "false",
                "temperature": "0.0",
                "top_p": "1.0",
                "top_k": "50",
                "parse_tables": "true"
            }
            logger.info(f"📤 Sending request with parameters: {data}")
            resp = client.post(self.api_url, files=files, data=data)
            logger.info(f"📥 API Response status: {resp.status_code}")
            
            if resp.status_code != 200:
                logger.error(f"❌ API request failed: {resp.status_code} - {resp.text}")
                resp.raise_for_status()
            
            result = resp.json()
            logger.info(f"✅ API call successful, response keys: {list(result.keys())}")
            return result
    
    def _ocr_image_to_html_transformers(self, image_path: str) -> str:
        """Use Transformers model to extract table HTML from image."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Load model
            model, tokenizer, processor = _load_transformers_model()
            
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Use proper message format as per Hugging Face guide
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": f"file://{image_path}"},
                    {"type": "text", "text": USER_PROMPT},
                ]},
            ]
            
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            # Process inputs as per guide
            inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
            inputs = inputs.to(model.device)
            
            # Generate with proper parameters
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=4096, do_sample=False)
            
            # Decode properly as per guide
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            
            return output_text[0]
            
        except Exception as e:
            raise RuntimeError(f"Failed to extract tables from image using Transformers: {e}")


# Convenience functions for backward compatibility
def extract_tables_from_image(image_path: str, api_url: str = None) -> Optional[str]:
    """
    Convenience function to extract table HTML from an image using API (with Transformers fallback).
    
    Args:
        image_path: Path to the image file
        api_url: Optional API endpoint URL (defaults to OCR_API_URL from config)
        
    Returns:
        Raw HTML string containing tables, or None if extraction failed
    """
    extractor = TableExtractor(api_url=api_url)
    return extractor.extract_tables_from_image(image_path)

def extract_tables_from_image_transformers(image_path: str) -> Optional[str]:
    """
    Convenience function to extract table HTML from an image using Transformers only.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Raw HTML string containing tables, or None if extraction failed
    """
    extractor = TableExtractor(api_url=None)  # Disable API to force Transformers
    return extractor._ocr_image_to_html_transformers(image_path)

def extract_tables_from_image_api_only(image_path: str, api_url: str = None) -> Optional[str]:
    """
    Convenience function to extract table HTML from an image using API only (no fallback).
    
    Args:
        image_path: Path to the image file
        api_url: Optional API endpoint URL (defaults to OCR_API_URL from config)
        
    Returns:
        Raw HTML string containing tables, or None if extraction failed
    """
    extractor = TableExtractor(api_url=api_url)
    return extractor._ocr_image_via_api(image_path)