"""
CAPTCHA Solving Tool
====================

Provides CAPTCHA solving capabilities using third-party services (2Captcha, Anti-Captcha)
with a human fallback mechanism when automated solving fails or is not configured.

Supported CAPTCHA types:
- reCAPTCHA v2
- reCAPTCHA v3
- hCaptcha

Configuration:
- Set TWOCAPTCHA_KEY in environment for 2Captcha service
- Set ANTICAPTCHA_KEY in environment for Anti-Captcha service (fallback)
"""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# Load environment variables from .env file
def _load_env():
    """Load .env file from multiple possible locations."""
    try:
        from dotenv import load_dotenv

        # Try multiple paths to find .env
        possible_paths = [
            # Relative to this file
            Path(__file__).parent.parent.parent / ".env",
            # Current working directory
            Path.cwd() / ".env",
            Path.cwd() / "maven2_fix" / ".env",
            # Absolute known path
            Path("/home/user/maven/maven2_fix/.env"),
        ]

        for env_path in possible_paths:
            if env_path.exists():
                load_dotenv(env_path, override=True)
                logging.getLogger(__name__).info(f"Loaded .env from {env_path}")
                return True

        logging.getLogger(__name__).warning("No .env file found in expected locations")
        return False
    except ImportError:
        logging.getLogger(__name__).warning("python-dotenv not installed, using system environment")
        return False

_load_env()

logger = logging.getLogger(__name__)


class CaptchaType(Enum):
    """Supported CAPTCHA types."""
    RECAPTCHA_V2 = "recaptcha_v2"
    RECAPTCHA_V3 = "recaptcha_v3"
    HCAPTCHA = "hcaptcha"
    UNKNOWN = "unknown"


class CaptchaSolverError(Exception):
    """Base exception for CAPTCHA solver errors."""
    pass


class CaptchaConfigError(CaptchaSolverError):
    """Raised when CAPTCHA API keys are not configured."""
    pass


class CaptchaTimeoutError(CaptchaSolverError):
    """Raised when CAPTCHA solving times out."""
    pass


class CaptchaSubmitError(CaptchaSolverError):
    """Raised when CAPTCHA submission fails."""
    pass


class HumanInterventionRequired(CaptchaSolverError):
    """Raised when human intervention is needed to solve CAPTCHA."""

    def __init__(self, url: str, reason: str = "Automated solver unavailable or failed"):
        self.url = url
        self.reason = reason
        super().__init__(
            f"[HUMAN NEEDED] CAPTCHA at {url} - please solve it manually and type DONE when ready. "
            f"Reason: {reason}"
        )


@dataclass
class CaptchaResult:
    """Result of a CAPTCHA solve attempt."""
    success: bool
    token: Optional[str] = None
    captcha_type: CaptchaType = CaptchaType.UNKNOWN
    service_used: Optional[str] = None
    solve_time_seconds: float = 0.0
    error: Optional[str] = None
    requires_human: bool = False


def get_captcha_api_key() -> Tuple[Optional[str], str]:
    """
    Get the CAPTCHA API key from environment.

    Returns:
        Tuple of (api_key, service_name) or (None, "") if not configured.
    """
    # Try 2Captcha first (generally cheaper and faster)
    twocaptcha_key = os.getenv("TWOCAPTCHA_KEY")
    if twocaptcha_key:
        return twocaptcha_key, "2captcha"

    # Fallback to Anti-Captcha
    anticaptcha_key = os.getenv("ANTICAPTCHA_KEY")
    if anticaptcha_key:
        return anticaptcha_key, "anticaptcha"

    return None, ""


def detect_captcha_type(sitekey: str) -> CaptchaType:
    """
    Detect the type of CAPTCHA based on sitekey format.

    Args:
        sitekey: The data-sitekey attribute value

    Returns:
        CaptchaType enum value
    """
    if not sitekey:
        return CaptchaType.UNKNOWN

    # hCaptcha sitekeys are typically UUIDs with dashes
    if "-" in sitekey and len(sitekey) == 36:
        return CaptchaType.HCAPTCHA

    # reCAPTCHA sitekeys are typically alphanumeric, ~40 chars
    if len(sitekey) >= 40:
        return CaptchaType.RECAPTCHA_V2  # Default to v2, can be overridden

    return CaptchaType.RECAPTCHA_V2


def solve_captcha(
    sitekey: str,
    url: str,
    proxy: Optional[str] = None,
    captcha_type: Optional[CaptchaType] = None,
    timeout_seconds: int = 180,
    poll_interval_seconds: int = 5,
) -> CaptchaResult:
    """
    Solve a CAPTCHA using 2Captcha or Anti-Captcha service.

    Args:
        sitekey: The data-sitekey attribute from the CAPTCHA element
        url: The page URL where the CAPTCHA appears
        proxy: Optional proxy in format "user:pass@host:port" or "host:port"
        captcha_type: Type of CAPTCHA (auto-detected if not specified)
        timeout_seconds: Maximum time to wait for solution (default 180s)
        poll_interval_seconds: Time between polling attempts (default 5s)

    Returns:
        CaptchaResult with token if successful

    Raises:
        HumanInterventionRequired: If automated solving is not available
        CaptchaTimeoutError: If solving times out
        CaptchaSubmitError: If submission to service fails
    """
    import requests

    start_time = time.time()

    # Get API key
    api_key, service = get_captcha_api_key()
    if not api_key:
        raise HumanInterventionRequired(
            url=url,
            reason="No CAPTCHA API key configured (set TWOCAPTCHA_KEY or ANTICAPTCHA_KEY)"
        )

    # Auto-detect captcha type if not specified
    if captcha_type is None:
        captcha_type = detect_captcha_type(sitekey)

    logger.info(f"[CAPTCHA] Solving {captcha_type.value} at {url} using {service}")

    # Determine service endpoint and method
    if service == "2captcha":
        result = _solve_with_2captcha(
            api_key=api_key,
            sitekey=sitekey,
            url=url,
            captcha_type=captcha_type,
            proxy=proxy,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )
    else:
        result = _solve_with_anticaptcha(
            api_key=api_key,
            sitekey=sitekey,
            url=url,
            captcha_type=captcha_type,
            proxy=proxy,
            timeout_seconds=timeout_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    result.solve_time_seconds = time.time() - start_time
    result.captcha_type = captcha_type
    result.service_used = service

    if result.success:
        logger.info(f"[CAPTCHA] Solved! Token length: {len(result.token or '')}")
    else:
        logger.warning(f"[CAPTCHA] Failed: {result.error}")

    return result


def _solve_with_2captcha(
    api_key: str,
    sitekey: str,
    url: str,
    captcha_type: CaptchaType,
    proxy: Optional[str],
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> CaptchaResult:
    """Solve CAPTCHA using 2Captcha service."""
    import requests

    base_url = "https://2captcha.com"

    # Determine method based on captcha type
    if captcha_type == CaptchaType.HCAPTCHA:
        method = "hcaptcha"
        key_param = "sitekey"
    else:
        method = "userrecaptcha"
        key_param = "googlekey"

    # Submit CAPTCHA
    payload = {
        "key": api_key,
        "method": method,
        key_param: sitekey,
        "pageurl": url,
        "json": 1,
    }

    if proxy:
        payload["proxy"] = proxy
        payload["proxytype"] = "HTTP"

    try:
        resp = requests.post(f"{base_url}/in.php", data=payload, timeout=30)
        resp_data = resp.json()
    except requests.RequestException as e:
        return CaptchaResult(
            success=False,
            error=f"Failed to submit CAPTCHA: {str(e)}",
            requires_human=True,
        )

    if resp_data.get("status") != 1:
        error_msg = resp_data.get("request", "Unknown error")
        return CaptchaResult(
            success=False,
            error=f"2Captcha submission failed: {error_msg}",
            requires_human=True,
        )

    captcha_id = resp_data["request"]
    logger.info(f"[CAPTCHA] Submitted to 2Captcha, ID: {captcha_id}, polling...")

    # Poll for result
    max_attempts = timeout_seconds // poll_interval_seconds
    for attempt in range(max_attempts):
        time.sleep(poll_interval_seconds)

        try:
            result = requests.get(
                f"{base_url}/res.php",
                params={"key": api_key, "action": "get", "id": captcha_id, "json": 1},
                timeout=30,
            ).json()
        except requests.RequestException as e:
            logger.warning(f"[CAPTCHA] Poll attempt {attempt + 1} failed: {e}")
            continue

        if result.get("status") == 1:
            return CaptchaResult(
                success=True,
                token=result["request"],
            )

        # Check if still processing
        if result.get("request") == "CAPCHA_NOT_READY":
            logger.debug(f"[CAPTCHA] Still solving... (attempt {attempt + 1}/{max_attempts})")
            continue

        # Some other error
        error_msg = result.get("request", "Unknown error")
        return CaptchaResult(
            success=False,
            error=f"2Captcha solving failed: {error_msg}",
            requires_human=True,
        )

    return CaptchaResult(
        success=False,
        error=f"CAPTCHA solving timed out after {timeout_seconds}s",
        requires_human=True,
    )


def _solve_with_anticaptcha(
    api_key: str,
    sitekey: str,
    url: str,
    captcha_type: CaptchaType,
    proxy: Optional[str],
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> CaptchaResult:
    """Solve CAPTCHA using Anti-Captcha service."""
    import requests

    base_url = "https://api.anti-captcha.com"

    # Determine task type
    if captcha_type == CaptchaType.HCAPTCHA:
        task_type = "HCaptchaTaskProxyless" if not proxy else "HCaptchaTask"
    else:
        task_type = "RecaptchaV2TaskProxyless" if not proxy else "RecaptchaV2Task"

    # Create task
    task = {
        "type": task_type,
        "websiteURL": url,
        "websiteKey": sitekey,
    }

    if proxy:
        # Parse proxy
        parts = proxy.split("@")
        if len(parts) == 2:
            user_pass, host_port = parts
            user, password = user_pass.split(":")
            host, port = host_port.split(":")
            task.update({
                "proxyType": "http",
                "proxyAddress": host,
                "proxyPort": int(port),
                "proxyLogin": user,
                "proxyPassword": password,
            })
        else:
            host, port = proxy.split(":")
            task.update({
                "proxyType": "http",
                "proxyAddress": host,
                "proxyPort": int(port),
            })

    try:
        resp = requests.post(
            f"{base_url}/createTask",
            json={"clientKey": api_key, "task": task},
            timeout=30,
        ).json()
    except requests.RequestException as e:
        return CaptchaResult(
            success=False,
            error=f"Failed to submit CAPTCHA: {str(e)}",
            requires_human=True,
        )

    if resp.get("errorId") != 0:
        error_msg = resp.get("errorDescription", "Unknown error")
        return CaptchaResult(
            success=False,
            error=f"Anti-Captcha submission failed: {error_msg}",
            requires_human=True,
        )

    task_id = resp["taskId"]
    logger.info(f"[CAPTCHA] Submitted to Anti-Captcha, task ID: {task_id}, polling...")

    # Poll for result
    max_attempts = timeout_seconds // poll_interval_seconds
    for attempt in range(max_attempts):
        time.sleep(poll_interval_seconds)

        try:
            result = requests.post(
                f"{base_url}/getTaskResult",
                json={"clientKey": api_key, "taskId": task_id},
                timeout=30,
            ).json()
        except requests.RequestException as e:
            logger.warning(f"[CAPTCHA] Poll attempt {attempt + 1} failed: {e}")
            continue

        if result.get("errorId") != 0:
            error_msg = result.get("errorDescription", "Unknown error")
            return CaptchaResult(
                success=False,
                error=f"Anti-Captcha solving failed: {error_msg}",
                requires_human=True,
            )

        if result.get("status") == "ready":
            solution = result.get("solution", {})
            token = solution.get("gRecaptchaResponse") or solution.get("token")
            return CaptchaResult(
                success=True,
                token=token,
            )

        logger.debug(f"[CAPTCHA] Still solving... (attempt {attempt + 1}/{max_attempts})")

    return CaptchaResult(
        success=False,
        error=f"CAPTCHA solving timed out after {timeout_seconds}s",
        requires_human=True,
    )


# Convenience function for use by browser actions
def solve_captcha_or_request_human(sitekey: str, url: str, proxy: Optional[str] = None) -> str:
    """
    Attempt to solve CAPTCHA, raising HumanInterventionRequired if it fails.

    This is the main entry point for browser automation.

    Args:
        sitekey: The data-sitekey from the CAPTCHA element
        url: Current page URL
        proxy: Optional proxy string

    Returns:
        The CAPTCHA token if solved successfully

    Raises:
        HumanInterventionRequired: If solving fails or is not configured
    """
    try:
        result = solve_captcha(sitekey, url, proxy)

        if result.success and result.token:
            return result.token

        # Solving failed, request human intervention
        raise HumanInterventionRequired(
            url=url,
            reason=result.error or "Automated solver failed"
        )

    except HumanInterventionRequired:
        raise
    except Exception as e:
        raise HumanInterventionRequired(
            url=url,
            reason=f"Unexpected error: {str(e)}"
        )
