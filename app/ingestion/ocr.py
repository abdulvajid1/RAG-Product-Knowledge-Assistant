import io
import os
import shutil
import logging
from typing import Tuple, List, Optional
from PIL import Image
import pytesseract
from pytesseract import Output

from app.config import get_settings

logger = logging.getLogger(__name__)

_rapid_ocr_instance = None


def get_rapid_ocr():
    """Lazily instantiate RapidOCR engine."""
    global _rapid_ocr_instance
    if _rapid_ocr_instance is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _rapid_ocr_instance = RapidOCR()
            logger.info("Initialized RapidOCR engine as zero-binary OCR provider.")
        except Exception as e:
            logger.debug(f"RapidOCR not available: {e}")
            _rapid_ocr_instance = False
    return _rapid_ocr_instance if _rapid_ocr_instance is not False else None


def find_tesseract_executable() -> Optional[str]:
    """Locate Tesseract binary from config, PATH, or standard installation directories."""
    settings = get_settings()
    if settings.tesseract_cmd and os.path.exists(settings.tesseract_cmd):
        return settings.tesseract_cmd

    # Check system PATH
    found = shutil.which("tesseract")
    if found:
        return found

    # Standard Windows install locations (including winget UB-Mannheim default)
    win_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%ProgramFiles%\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%ProgramFiles(x86)%\Tesseract-OCR\tesseract.exe"),
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


def is_ocr_available() -> bool:
    """Return True if any OCR engine (Tesseract or RapidOCR) is available."""
    return is_tesseract_available() or (get_rapid_ocr() is not None)


def ocr_image_bytes(
    image_bytes: bytes,
    low_conf_word_cutoff: Optional[float] = None,
    low_conf_chunk_cutoff: Optional[float] = None,
) -> Tuple[str, Optional[float], bool]:
    """Run OCR on raw image bytes."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return ocr_image(img, low_conf_word_cutoff, low_conf_chunk_cutoff)
    except Exception as e:
        logger.error(f"Failed to open or decode image bytes for OCR: {e}")
        return "", None, True


def ocr_pixmap(
    pixmap,
    low_conf_word_cutoff: Optional[float] = None,
    low_conf_chunk_cutoff: Optional[float] = None,
) -> Tuple[str, Optional[float], bool]:
    """Run OCR on a PyMuPDF Pixmap."""
    try:
        img = Image.open(io.BytesIO(pixmap.tobytes("png")))
        return ocr_image(img, low_conf_word_cutoff, low_conf_chunk_cutoff)
    except Exception as e:
        logger.error(f"Failed to convert PyMuPDF pixmap to image for OCR: {e}")
        return "", None, True


def _ocr_with_tesseract(
    image: Image.Image,
    word_cutoff: float,
    chunk_cutoff: float,
) -> Tuple[str, Optional[float], bool]:
    """Execute OCR using Tesseract."""
    cmd = find_tesseract_executable()
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
        logger.warning("Tesseract OCR yielded no text above word-confidence threshold.")
        return "", None, True

    extracted_text = "\n".join(kept_words).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    is_low_conf = mean_conf < chunk_cutoff

    if is_low_conf:
        logger.warning(
            f"Extracted Tesseract OCR text has low mean confidence: {mean_conf:.1f}% (threshold: {chunk_cutoff}%)"
        )

    return extracted_text, round(mean_conf, 2), is_low_conf


def _ocr_with_rapidocr(
    image: Image.Image,
    word_cutoff: float,
    chunk_cutoff: float,
) -> Tuple[str, Optional[float], bool]:
    """Execute OCR using RapidOCR (ONNX)."""
    engine = get_rapid_ocr()
    if not engine:
        return "", None, True

    try:
        if image.mode != "RGB":
            image = image.convert("RGB")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        result, _ = engine(buf.getvalue())
    except Exception as e:
        logger.error(f"Failed to execute RapidOCR on image: {e}")
        return "", None, True

    if not result:
        logger.warning("RapidOCR detected no text blocks in image.")
        return "", None, True

    kept_lines: List[str] = []
    confidences: List[float] = []

    for item in result:
        # RapidOCR format: [box, text, score]
        text = str(item[1]).strip()
        try:
            score = float(item[2])
            conf = score * 100.0 if score <= 1.0 else score
        except (ValueError, TypeError):
            conf = 0.0

        if not text:
            continue

        if conf >= word_cutoff:
            kept_lines.append(text)
            confidences.append(conf)
        else:
            logger.debug(f"Dropping low-confidence RapidOCR line '{text}' (confidence={conf:.1f} < {word_cutoff})")

    if not kept_lines:
        logger.warning("RapidOCR yielded no lines above confidence threshold.")
        return "", None, True

    extracted_text = "\n".join(kept_lines).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    is_low_conf = mean_conf < chunk_cutoff

    if is_low_conf:
        logger.warning(
            f"Extracted RapidOCR text has low mean confidence: {mean_conf:.1f}% (threshold: {chunk_cutoff}%)"
        )

    return extracted_text, round(mean_conf, 2), is_low_conf


def ocr_image(
    image: Image.Image,
    low_conf_word_cutoff: Optional[float] = None,
    low_conf_chunk_cutoff: Optional[float] = None,
) -> Tuple[str, Optional[float], bool]:
    """
    Run OCR on a PIL Image with per-word/line confidence scoring.
    Dispatches to Tesseract if available, or RapidOCR as fallback.

    Returns:
        (extracted_text, mean_confidence, is_low_confidence)
    """
    settings = get_settings()
    word_cutoff = low_conf_word_cutoff or settings.ocr_low_conf_word
    chunk_cutoff = low_conf_chunk_cutoff or settings.ocr_low_conf_chunk

    if is_tesseract_available():
        return _ocr_with_tesseract(image, word_cutoff, chunk_cutoff)
    elif get_rapid_ocr() is not None:
        return _ocr_with_rapidocr(image, word_cutoff, chunk_cutoff)
    else:
        logger.warning(
            "No OCR engine is available (neither Tesseract nor RapidOCR). "
            "To enable OCR, install Tesseract via 'winget install UB-Mannheim.TesseractOCR' "
            "or pip install rapidocr-onnxruntime."
        )
        return "", None, True
