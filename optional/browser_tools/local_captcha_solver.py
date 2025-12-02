"""
Local CAPTCHA Solver - Placeholder
==================================

TODO: Implement local CAPTCHA solving.

This tool would provide CAPTCHA solving without external services.

Options to consider:
1. Tesseract OCR (pytesseract) - for simple text CAPTCHAs
2. ML models - for image-based CAPTCHAs
3. Audio solving - for audio CAPTCHAs

Dependencies (pick based on approach):
- pytesseract + Tesseract-OCR
- torch/tensorflow for ML models
- speech_recognition for audio

Usage:
    from optional.browser_tools.local_captcha_solver import solve_captcha

    solution = await solve_captcha(
        image_base64="...",
        captcha_type="text"  # or "recaptcha", "hcaptcha", etc.
    )

Implementation notes:
- Text CAPTCHAs: Use Tesseract OCR with preprocessing
- Image CAPTCHAs: Consider training a custom model
- reCAPTCHA v2: Would need browser automation + image recognition
- hCaptcha: Similar to reCAPTCHA

Limitations:
- Local solving is less reliable than cloud services
- Modern CAPTCHAs (v3, invisible) may not be solvable locally
- May require significant GPU resources for ML approaches
"""

import base64
from pathlib import Path
from typing import Optional


async def solve_captcha(
    image_base64: str,
    captcha_type: str = "text"
) -> Optional[str]:
    """
    Solve a CAPTCHA locally.

    Args:
        image_base64: Base64-encoded image of the CAPTCHA
        captcha_type: Type of CAPTCHA ("text", "image", "recaptcha", "hcaptcha")

    Returns:
        The solution string, or None if unsolvable

    Raises:
        NotImplementedError: This is a placeholder
    """
    raise NotImplementedError(
        "Local CAPTCHA solver not implemented. "
        "See docstring for implementation options."
    )


def solve_text_captcha_ocr(image_path: Path) -> Optional[str]:
    """
    Solve a simple text CAPTCHA using Tesseract OCR.

    Requires: pip install pytesseract
    And: Tesseract-OCR installed on system

    Args:
        image_path: Path to the CAPTCHA image

    Returns:
        Extracted text, or None if failed
    """
    raise NotImplementedError(
        "Install pytesseract and implement OCR-based solving"
    )
