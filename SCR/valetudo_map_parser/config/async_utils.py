"""Async utility functions for making NumPy and PIL operations truly async."""

import asyncio
import io
from typing import Any, Callable

import numpy as np
from numpy import rot90
from PIL import Image


async def make_async(func: Callable, *args, **kwargs) -> Any:
    """Convert a synchronous function to async by yielding control to the event loop."""
    return await asyncio.to_thread(func, *args, **kwargs)


class AsyncNumPy:
    """Async wrappers for NumPy operations that yield control to the event loop."""

    @staticmethod
    async def async_copy(array: np.ndarray) -> np.ndarray:
        """Async array copying."""
        return await make_async(np.copy, array)

    @staticmethod
    async def async_full(
        shape: tuple, fill_value: Any, dtype: np.dtype = None
    ) -> np.ndarray:
        """Async array creation with fill value."""
        return await make_async(np.full, shape, fill_value, dtype=dtype)

    @staticmethod
    async def async_rot90(array: np.ndarray, k: int = 1) -> np.ndarray:
        """Async array rotation."""
        return await make_async(rot90, array, k)


class AsyncPIL:
    """Async wrappers for PIL operations that yield control to the event loop."""

    @staticmethod
    async def async_fromarray(array: np.ndarray, mode: str = "RGBA") -> Image.Image:
        """Async PIL Image creation from NumPy array."""
        return await make_async(Image.fromarray, array, mode)

    @staticmethod
    async def async_resize(
        image: Image.Image, size: tuple, resample: int = None
    ) -> Image.Image:
        """Async image resizing."""
        if resample is None:
            resample = Image.LANCZOS
        return await make_async(image.resize, size, resample)

    @staticmethod
    async def async_save_to_bytes(
        image: Image.Image, format_type: str = "WEBP", **kwargs
    ) -> bytes:
        """Async image saving to bytes."""

        def save_to_bytes():
            buffer = io.BytesIO()
            image.save(buffer, format=format_type, **kwargs)
            return buffer.getvalue()

        return await make_async(save_to_bytes)


class AsyncParallel:
    """Helper functions for parallel processing with asyncio.gather()."""

    @staticmethod
    async def parallel_data_preparation(*tasks):
        """Execute multiple data preparation tasks in parallel."""
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    async def parallel_array_operations(base_array: np.ndarray, operations: list):
        """Execute multiple array operations in parallel on copies of the base array."""

        # Create tasks for parallel execution
        tasks = []
        for operation_func, *args in operations:
            # Each operation works on a copy of the base array
            array_copy = await AsyncNumPy.async_copy(base_array)
            tasks.append(operation_func(array_copy, *args))

        # Execute all operations in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions and return successful results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        return successful_results
