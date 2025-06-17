"""Module for preparing rate limiter recipes."""

import os

from langchain_core.rate_limiters import InMemoryRateLimiter

from clilm.utils import make_custom_parameter


class RateLimiterType:
    """Base class for rate limiter recipes."""

    rate_limiter: InMemoryRateLimiter

    def get(self) -> InMemoryRateLimiter:
        """Returns initialized rate limiter."""
        return self.rate_limiter


class NoRateLimiter(RateLimiterType):
    """Recipe for no rate limiting."""

    rate_limiter = None


class Memory(RateLimiterType):
    """Recipe for in-memory rate limiter.

    Warning:
        Implements langchain's example rate limiter, which is very slow!
    """

    def __init__(
        self,
        requests_per_second: float = 0.1,
        check_every_n_seconds: float = 0.1,
        max_bucket_size: int = 10,
        **kwargs,
    ):
        self.rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=check_every_n_seconds,
            max_bucket_size=max_bucket_size,
            **kwargs,
        )


### add new classes above this line ###
name = os.path.basename(__file__)
types = {
    k: v for k, v in locals().items() if getattr(v, "__base__", None) == RateLimiterType
}
PARAMETER = make_custom_parameter(name, types)
