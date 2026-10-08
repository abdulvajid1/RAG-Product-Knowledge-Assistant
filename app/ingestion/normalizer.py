import re
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters to standard NFKC form."""
    return unicodedata.normalize("NFKC", text)


def fix_hyphenated_linebreaks(text: str) -> str:
    """
    Rejoin words split across lines by hyphens.
    E.g. 'operat-\ning temperature' -> 'operating temperature'
    """
    return re.sub(r'(\b\w+)-\s*\n\s*(\w+\b)', r'\1\2', text)


def normalize_units(text: str) -> str:
    """
    Standardize common units and degree representations.
    E.g. '85 °C' -> '85°C', '85 deg C' -> '85°C', '550 W' -> '550W'
    """
    # Normalize degrees and temperature
    text = re.sub(r'(\d+)\s*(?:°|deg|degrees)\s*C\b', r'\1°C', text, flags=re.IGNORECASE)
    text = re.sub(r'(\d+)\s*(?:°|deg|degrees)\s*F\b', r'\1°F', text, flags=re.IGNORECASE)

    # Normalize power, voltage, current
    text = re.sub(r'(\d+)\s*W\b', r'\1W', text)
    text = re.sub(r'(\d+)\s*kW\b', r'\1kW', text)
    text = re.sub(r'(\d+)\s*V\b', r'\1V', text)
    text = re.sub(r'(\d+)\s*kV\b', r'\1kV', text)
    text = re.sub(r'(\d+)\s*mA\b', r'\1mA', text)
    text = re.sub(r'(\d+)\s*bar\b', r'\1 bar', text, flags=re.IGNORECASE)

    return text


def clean_whitespace(text: str) -> str:
    """Normalize irregular spaces and excessive line breaks."""
    # Replace non-breaking spaces and tabs with standard space
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)
    # Collapse 3 or more newlines into double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def strip_repeated_headers_footers(text: str) -> str:
    """Remove boilerplate page headers or footers like 'Page X of Y'."""
    text = re.sub(r'(?i)\bpage\s+\d+(\s+of\s+\d+)?\b', '', text)
    text = re.sub(r'[-=]{5,}', '---', text)
    return text


def clean_text(text: str) -> str:
    """Apply complete cleaning pipeline."""
    if not text:
        return ""
    text = normalize_unicode(text)
    text = fix_hyphenated_linebreaks(text)
    text = normalize_units(text)
    text = strip_repeated_headers_footers(text)
    text = clean_whitespace(text)
    return text

