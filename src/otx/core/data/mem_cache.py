# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Memory cache handler implementations and singleton class to call them."""
from __future__ import annotations

import ctypes as ct
import logging
import multiprocessing as mp
import re
import signal
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import psutil

if TYPE_CHECKING:
    from multiprocessing.managers import DictProxy
    from multiprocessing.synchronize import Lock

logger = logging.getLogger()

GIB = 1024**3

__all__ = [
    "MemCacheHandlerSingleton",
    "MemCacheHandlerError",
    "parse_mem_cache_size_to_int",
]


def parse_mem_cache_size_to_int(mem_cache_size: str) -> int:
    """Parse memory size string to integer.

    For example, "2GB" => 2,000,000,000 or "3MIB" => 3 * 1024 * 1024.
    """
    match = re.match(r"^([\d\.]+)\s*([a-zA-Z]{0,3})$", mem_cache_size.strip())

    if match is None:
        msg = f"Cannot parse {mem_cache_size} string."
        raise ValueError(msg)

    units = {
        "": 1,
        "B": 1,
        "KIB": 2**10,
        "MIB": 2**20,
        "GIB": 2**30,
        "KB": 10**3,
        "MB": 10**6,
        "GB": 10**9,
        "K": 2**10,
        "M": 2**20,
        "G": 2**30,
    }

    number, unit = int(match.group(1)), match.group(2).upper()

    if unit not in units:
        msg = f"{mem_cache_size} has disallowed unit ({unit})."
        raise ValueError(msg)

    return number * units[unit]


class _DummyLock:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class MemCacheHandlerBase:
    """Base class for memory cache handler.

    It will be combined with LoadImageFromOTXDataset to store/retrieve the samples in memory.
    """

    def __init__(self, mem_size: int):
        self._init_data_structs(mem_size)

    def _init_data_structs(self, mem_size: int) -> None:
        self._arr = (ct.c_uint8 * mem_size)()
        self._cur_page = ct.c_size_t(0)
        self._cache_addr: dict | DictProxy = {}
        self._lock: Lock | _DummyLock = _DummyLock()
        self._freeze = ct.c_bool(False)

    def __len__(self) -> int:
        """Get the number of cached items."""
        return len(self._cache_addr)

    @property
    def mem_size(self) -> int:
        """Get the reserved memory pool size (bytes)."""
        return len(self._arr)

    def get(self, key: Any) -> tuple[np.ndarray | None, dict | None]:  # noqa: ANN401
        """Try to look up the cached item with the given key.

        Args:
            key (Any): A key for looking up the cached item

        Returns:
            If succeed return (np.ndarray, Dict), otherwise return (None, None)
        """
        if self.mem_size == 0 or key not in self._cache_addr:
            return None, None

        addr = self._cache_addr[key]

        offset, count, dtype, shape, strides, meta = addr

        data = np.frombuffer(self._arr, dtype=dtype, count=count, offset=offset)
        return np.lib.stride_tricks.as_strided(data, shape, strides), meta

    def put(
        self,
        key: Any,  # noqa: ANN401
        data: np.ndarray,
        meta: dict | None = None,
    ) -> int | None:
        """Try to store np.ndarray and metadata with a key to the reserved memory pool.

        Args:
            key (Any): A key to store the cached item
            data (np.ndarray): A data sample to store
            meta (Optional[Dict]): A metadata of the data sample

        Returns:
            Optional[int]: If succeed return the address of cached item in memory pool
        """
        if self._freeze.value:
            return None

        data_bytes = data.size * data.itemsize

        with self._lock:
            new_page = self._cur_page.value + data_bytes

            if key in self._cache_addr or new_page > self.mem_size:
                return None

            offset = ct.byref(self._arr, self._cur_page.value)
            ct.memmove(offset, data.ctypes.data, data_bytes)

            self._cache_addr[key] = (
                self._cur_page.value,
                data.size,
                data.dtype,
                data.shape,
                data.strides,
                meta,
            )
            self._cur_page.value = new_page
            return new_page

    def __repr__(self) -> str:
        """Representation for the current handler status."""
        perc = (
            100.0 * self._cur_page.value / self.mem_size if self.mem_size > 0 else 0.0
        )
        return (
            f"{self.__class__.__name__} "
            f"uses {self._cur_page.value} / {self.mem_size} ({perc:.1f}%) memory pool and "
            f"store {len(self)} items."
        )

    def freeze(self) -> None:
        """If frozen, it is impossible to store a new item anymore."""
        self._freeze.value = True

    def unfreeze(self) -> None:
        """If unfrozen, it is possible to store a new item."""
        self._freeze.value = False


class MemCacheHandlerForSP(MemCacheHandlerBase):
    """Memory caching handler for single processing.

    Use if PyTorch's DataLoader.num_workers == 0.
    """


class MemCacheHandlerForMP(MemCacheHandlerBase):
    """Memory caching handler for multi processing.

    Use if PyTorch's DataLoader.num_workers > 0.
    """

    def _init_data_structs(self, mem_size: int) -> None:
        self._arr = mp.Array(ct.c_uint8, mem_size, lock=False)
        self._cur_page = mp.Value(ct.c_size_t, 0, lock=False)

        self._manager = mp.Manager()
        self._cache_addr: DictProxy = self._manager.dict()
        self._lock = mp.Lock()
        self._freeze = mp.Value(ct.c_bool, False, lock=False)

    def __del__(self) -> None:
        """When deleting, manager should also be shutdowned."""
        self._manager.shutdown()


class MemCacheHandlerError(Exception):
    """Exception class for MemCacheHandler."""


class MemCacheHandlerSingleton:
    """A singleton class to create, delete and get MemCacheHandlerBase."""

    instance: MemCacheHandlerBase
    is_shutdown: bool = False
    CPU_MEM_LIMITS_GIB: int = 30

    @classmethod
    def get(cls) -> MemCacheHandlerBase | None:
        """Get the created MemCacheHandlerBase.

        If no one is created before, raise RuntimeError.
        """
        if cls.is_shutdown:
            # Gracefully return None
            return None

        if not hasattr(cls, "instance"):
            cls_name = cls.__name__
            msg = f"Before calling {cls_name}.get(), you should call {cls_name}.create() first."
            warnings.warn(message=msg, stacklevel=1)
            return None

        return cls.instance

    @classmethod
    def create(cls, mode: str, mem_size: int) -> MemCacheHandlerBase:
        """Create a new MemCacheHandlerBase instance.

        Args:
            mode (str): There are two options: null, multiprocessing or singleprocessing.
            mem_size (int): The size of memory pool (bytes).
        """
        # COPY FROM mmcv.runner.get_dist_info
        from torch import distributed

        world_size = (
            distributed.get_world_size()
            if distributed.is_available() and distributed.is_initialized()
            else 1
        )

        # Prevent CPU OOM issue
        memory_info = psutil.virtual_memory()
        available_cpu_mem = memory_info.available / GIB

        if world_size > 1:
            mem_size = mem_size // world_size
            available_cpu_mem = available_cpu_mem // world_size
            logger.info(
                f"Since world_size={world_size} > 1, each worker a {mem_size} size memory pool.",
            )

        logger.info(f"Try to create a {mem_size} size memory pool.")
        if available_cpu_mem < ((mem_size / GIB) + cls.CPU_MEM_LIMITS_GIB):
            logger.warning("No available CPU memory left, mem_size will be set to 0.")
            mem_size = 0

        if mode == "null" or mem_size == 0:
            cls.instance = MemCacheHandlerBase(mem_size=0)
            cls.instance.freeze()
        elif mode == "multiprocessing":
            cls.instance = MemCacheHandlerForMP(mem_size)
        elif mode == "singleprocessing":
            cls.instance = MemCacheHandlerForSP(mem_size)
        else:
            msg = f"{mode} is unknown mode."
            raise MemCacheHandlerError(msg)

        # Should delete if receive sigint to gracefully terminate
        original_handler = signal.getsignal(signal.SIGINT)

        def _new_handler(signum, frame) -> None:  # noqa: ANN001
            original_handler(signum, frame)  # type: ignore[operator, misc]
            cls.shutdown()

        signal.signal(signal.SIGINT, _new_handler)

        return cls.instance

    @classmethod
    def delete(cls) -> None:
        """Delete the existing MemCacheHandlerBase instance."""
        if hasattr(cls, "instance"):
            del cls.instance

    @classmethod
    def shutdown(cls) -> None:
        """This method will be called if the process terminates by the interrupt signal."""
        cls.is_shutdown = True
        cls.delete()
