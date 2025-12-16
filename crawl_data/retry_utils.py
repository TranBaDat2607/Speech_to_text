#!/usr/bin/env python3
"""
Retry Utilities
Provides retry decorators and utilities for handling transient failures
"""

import time
import logging
from functools import wraps
from typing import Callable, Type, Tuple

logger = logging.getLogger(__name__)


def retry_on_exception(
    max_attempts: int = 3,
    delay: float = 2.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    logger_instance: logging.Logger = None
):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
        exceptions: Tuple of exception types to catch and retry
        logger_instance: Logger instance for logging retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger_instance or logger
            current_delay = delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        _logger.error(
                            f"Function {func.__name__} failed after {max_attempts} attempts: {e}"
                        )
                        raise

                    _logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {e}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

            return None

        return wrapper
    return decorator


class RetryableOperation:
    """Context manager for retryable operations with better error tracking"""

    def __init__(self, operation_name: str, max_attempts: int = 3,
                 delay: float = 2.0, backoff: float = 2.0):
        self.operation_name = operation_name
        self.max_attempts = max_attempts
        self.delay = delay
        self.backoff = backoff
        self.attempt = 0
        self.errors = []

    def __enter__(self):
        self.attempt += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.errors.append({
                'attempt': self.attempt,
                'error': str(exc_val),
                'type': exc_type.__name__
            })

            if self.attempt < self.max_attempts:
                current_delay = self.delay * (self.backoff ** (self.attempt - 1))
                logger.warning(
                    f"{self.operation_name} - Attempt {self.attempt}/{self.max_attempts} failed: {exc_val}. "
                    f"Retrying in {current_delay:.1f}s..."
                )
                time.sleep(current_delay)
                return True  # Suppress exception to allow retry
            else:
                logger.error(
                    f"{self.operation_name} - All {self.max_attempts} attempts failed. "
                    f"Errors: {self.errors}"
                )
                return False  # Let exception propagate
        return False


def safe_execute(func: Callable, *args, default=None, log_errors: bool = True, **kwargs):
    """
    Safely execute a function and return default value on error

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default: Default value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error executing {func.__name__}: {e}", exc_info=True)
        return default


class OperationTracker:
    """Track success/failure of operations for reporting"""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.total_attempts = 0
        self.successful = 0
        self.failed = 0
        self.failures = []

    def record_success(self, item_id: str = None):
        """Record a successful operation"""
        self.total_attempts += 1
        self.successful += 1
        if item_id:
            logger.info(f"{self.operation_name} succeeded for {item_id}")

    def record_failure(self, item_id: str = None, error: str = None):
        """Record a failed operation"""
        self.total_attempts += 1
        self.failed += 1
        failure_record = {'item_id': item_id, 'error': error}
        self.failures.append(failure_record)
        if item_id:
            logger.error(f"{self.operation_name} failed for {item_id}: {error}")

    def get_summary(self) -> dict:
        """Get summary of operations"""
        return {
            'operation': self.operation_name,
            'total_attempts': self.total_attempts,
            'successful': self.successful,
            'failed': self.failed,
            'success_rate': f"{(self.successful / self.total_attempts * 100):.1f}%" if self.total_attempts > 0 else "0%",
            'failures': self.failures
        }

    def print_summary(self):
        """Print summary to console"""
        summary = self.get_summary()
        print(f"\n{self.operation_name} Summary:")
        print(f"  Total: {summary['total_attempts']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']}")
        if self.failures:
            print(f"  Failed items: {len(self.failures)}")
