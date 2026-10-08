
import os
import shutil
import logging
from typing import Tuple, List, Optional
from PIL import Image
import pytesseract
from pytesseract import Output

from app.config import get_settings

logger = logging.getLogger(__name__)


def find_tesseract_executable() -> Optional[str]:
    """Locate Tesseract binary from config, PATH, or standard installation directories."""
    settings = get_settings()
    if settings.tesseract_cmd and os.path.exists(settings.tesseract_cmd):
        return settings.tesseract_cmd

    # Check system PATH
    found = shutil.which("tesseract")
    if found:
        return found

    # Standard Windows install locations
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for p in win_paths:
        if os.path.exists(p):
            return p

    return None


def is_tesseract_available() -> bool:
    """Return True if Tesseract OCR binary is located and usable."""
    cmd = find_tesseract_executable()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        return True
    return False


def ocr_image(
    image: Image.Image,
    low_conf_word_cutoff: Optional[float] = None,
    low_conf_chunk_cutoff: Optional[float] = None,
) -> Tuple[str, Optional[float], bool]:
    """
    Run OCR on a PIL Image with per-word confidence scoring.

    Returns:
        (extracted_text, mean_confidence, is_low_confidence)
    """
    settings = get_settings()
    word_cutoff = low_conf_word_cutoff or settings.ocr_low_conf_word
    chunk_cutoff = low_conf_chunk_cutoff or settings.ocr_low_conf_chunk

    cmd = find_tesseract_executable()
    if not cmd:
        logger.warning("Tesseract OCR is not installed or not found on system path.")
        return "", None, True

    pytesseract.pytesseract.tesseract_cmd = cmd

    try:
        data = pytesseract.image_to_data(image, output_type=Output.DICT)
    except Exception as e:
        logger.error(f"Failed to execute Tesseract OCR on image: {e}")
        return "", None, True

    n_boxes = len(data["text"])
    kept_words: List[str] = []
    confidences: List[float] = []

    current_line = []
    current_line_num = -1

    for i in range(n_boxes):
        text = data["text"][i].strip()
        conf_raw = data["conf"][i]

        try:
            conf = float(conf_raw)
        except (ValueError, TypeError):
            conf = -1.0

        # pytesseract returns -1 for empty/whitespace bounding boxes
        if not text or conf < 0:
            continue

        if conf >= word_cutoff:
            confidences.append(conf)
            line_num = data["line_num"][i]
            if current_line_num != line_num and current_line:
                kept_words.append(" ".join(current_line))
                current_line = [text]
                current_line_num = line_num
            else:
                current_line.append(text)
                current_line_num = line_num
        else:
            logger.debug(f"Dropping low-confidence OCR word '{text}' (confidence={conf} < {word_cutoff})")

    if current_line:
        kept_words.append(" ".join(current_line))

    if not kept_words:
        logger.warning("OCR yielded no text above word-confidence threshold.")
        return "", None, True

    extracted_text = "\n".join(kept_words).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    is_low_conf = mean_conf < chunk_cutoff

    if is_low_conf:
        logger.warning(
            f"Extracted OCR text has low mean confidence: {mean_conf:.1f}% (threshold: {chunk_cutoff}%)"
        )

    return extracted_text, round(mean_conf, 2), is_low_conf

