"""
Browser Rate Limiter
====================

Rate limiting per domain to prevent overloading target sites.
Uses a token bucket algorithm per domain.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

from optional.browser_runtime.config import get_config


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    rate: float  # Tokens per second
    capacity: int  # Max tokens
    tokens: float = field(default=0)
    last_update: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def try_consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were consumed, False if rate limited
        """
        now = time.time()
        elapsed = now - self.last_update

        # Refill bucket
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

        # Try to consume
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    async def wait_for_token(self, tokens: int = 1) -> None:
        """
        Wait until tokens are available, then consume.

        Args:
            tokens: Number of tokens to consume
        """
        while not self.try_consume(tokens):
            # Calculate wait time
            needed = tokens - self.tokens
            wait_time = needed / self.rate
            await asyncio.sleep(min(wait_time, 1.0))


class DomainRateLimiter:
    """Rate limiter that tracks requests per domain."""

    def __init__(
        self,
        requests_per_second: Optional[float] = None,
        burst_capacity: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Rate limit per domain (requests/second)
            burst_capacity: Maximum burst capacity
        """
        config = get_config()

        self.rate = requests_per_second or (config.rate_limit_per_domain / 60.0)  # Convert from per-minute
        self.capacity = burst_capacity or config.rate_limit_per_domain

        self._buckets: Dict[str, TokenBucket] = {}
        self._lock = asyncio.Lock()

    def _get_bucket(self, domain: str) -> TokenBucket:
        """Get or create bucket for domain."""
        if domain not in self._buckets:
            self._buckets[domain] = TokenBucket(
                rate=self.rate,
                capacity=self.capacity
            )
        return self._buckets[domain]

    def check_rate_limit(self, domain: str) -> bool:
        """
        Check if request to domain is allowed.

        Args:
            domain: Target domain

        Returns:
            True if request is allowed, False if rate limited
        """
        bucket = self._get_bucket(domain)
        return bucket.try_consume(1)

    async def acquire(self, domain: str) -> None:
        """
        Acquire permission to make a request to domain.
        Waits if rate limited.

        Args:
            domain: Target domain
        """
        async with self._lock:
            bucket = self._get_bucket(domain)

        await bucket.wait_for_token(1)

    def get_wait_time(self, domain: str) -> float:
        """
        Get estimated wait time for next request.

        Args:
            domain: Target domain

        Returns:
            Seconds until next request is allowed (0 if immediately allowed)
        """
        bucket = self._get_bucket(domain)

        # Update bucket
        now = time.time()
        elapsed = now - bucket.last_update
        available = min(bucket.capacity, bucket.tokens + elapsed * bucket.rate)

        if available >= 1:
            return 0.0

        needed = 1 - available
        return needed / bucket.rate

    def reset_domain(self, domain: str) -> None:
        """
        Reset rate limiter for a specific domain.

        Args:
            domain: Domain to reset
        """
        if domain in self._buckets:
            del self._buckets[domain]

    def reset_all(self) -> None:
        """Reset all rate limiters."""
        self._buckets.clear()


# Global rate limiter instance
_rate_limiter: Optional[DomainRateLimiter] = None


def get_rate_limiter() -> DomainRateLimiter:
    """Get the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = DomainRateLimiter()
    return _rate_limiter


def reset_rate_limiter() -> None:
    """Reset the global rate limiter."""
    global _rate_limiter
    _rate_limiter = None
