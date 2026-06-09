import torch
from typing import Tuple, Any, Optional, Sequence
from itertools import accumulate
from operator import mul


class MulticastHandle:
    def __init__(
        self,
        multicomm,
        size: Tuple[int],
        dtype: torch.dtype = None,
    ):
        super().__init__()
        # init
        self.rank_ = multicomm.GetRank()
        self.world_size_ = multicomm.GetWorldSize()

        # data buffer
        numel = list(accumulate(size, func=mul))[-1]
        element_size = dtype.itemsize
        buffer_size = numel * element_size
        self.buffer_size_ = buffer_size

        # signal pad buffer
        signal_offset = (buffer_size + 15) // 16 * 16

        def get_signal_size():
            MAX_CUDA_P2P_DOMAIN_SIZE = 72  # NVIDIA GB200 NVL72
            max_num_blocks = torch.cuda.get_device_properties(0).multi_processor_count
            # Maximally, a rank will need to sync with all other ranks, over all
            # channels. Each signal is 32 bits, which is the minimum unit for atomic cas.
            SIGNAL_SIZE = MAX_CUDA_P2P_DOMAIN_SIZE * max_num_blocks * 4
            return SIGNAL_SIZE

        self.signal_size_ = get_signal_size()
        total_size = signal_offset + self.signal_size_

        # create buffer
        self.org_buffer_dict_ = multicomm.CreateTensorSync(total_size)

        # memset 0
        self.org_buffer_dict_[self.rank][:] = 0

        # slice buffer
        self.data_buffer_list_ = [
            self.org_buffer_dict_[i][: self.buffer_size_] for i in range(self.world_size)
        ]
        self.multimem_data_buffer_ = self.org_buffer_dict_[-1][: self.buffer_size_]
        self.signal_buffer_list_ = [
            self.org_buffer_dict_[i][signal_offset:] for i in range(self.world_size)
        ]
        self.multimem_signal_buffer_ = self.org_buffer_dict_[-1][signal_offset:]
        for i in range(self.world_size):
            assert (
                self.signal_buffer_list_[i].data_ptr()
                == self.org_buffer_dict_[i].data_ptr() + signal_offset
            )

        # store all rank ptrs into a tensor
        self.data_buffer_ptrs_ = torch.empty(self.world_size, dtype=torch.int64, device="cpu")
        self.signal_buffer_ptrs_ = torch.empty(self.world_size, dtype=torch.int64, device="cpu")
        for i in range(self.world_size):
            self.data_buffer_ptrs_[i] = self.data_buffer_list_[i].data_ptr()
            self.signal_buffer_ptrs_[i] = self.signal_buffer_list_[i].data_ptr()
        device = self.org_buffer_dict_[self.rank].device
        self.data_buffer_ptrs_dev_ = self.data_buffer_ptrs_.to(device=device)
        self.signal_buffer_ptrs_dev_ = self.signal_buffer_ptrs_.to(device=device)

    @property
    def rank(self) -> int:
        return self.rank_

    @property
    def world_size(self) -> int:
        return self.world_size_

    @property
    def buffer_size(self) -> int:
        return self.buffer_size_

    @property
    def signal_size(self) -> int:
        return self.signal_size_

    @property
    def data_buffer_ptrs(self) -> torch.Tensor:
        return self.data_buffer_ptrs_

    @property
    def signal_buffer_ptrs(self) -> torch.Tensor:
        return self.signal_buffer_ptrs_

    @property
    def data_buffer_ptrs_dev(self) -> torch.Tensor:
        return self.data_buffer_ptrs_dev_

    @property
    def signal_buffer_ptrs_dev(self) -> torch.Tensor:
        return self.signal_buffer_ptrs_dev_

    def get_buffer(
        self, rank: int, *sizes: Any, dtype: Optional[torch.dtype] = None, storage_offset: int = 0
    ) -> torch.Tensor:
        assert 0 <= rank <= self.world_size

        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            sizes = tuple(sizes[0])
        else:
            sizes = tuple(sizes)

        if dtype is None:
            dtype = torch.get_default_dtype()

        numel = list(accumulate(sizes, func=mul))[-1]
        ask_size = numel * dtype.itemsize

        assert (
            storage_offset + ask_size <= self.buffer_size
        ), f"The requested buffer size(got {storage_offset} + {ask_size} = {storage_offset + ask_size}) exceeds the size of the hold buffer(got {self.buffer_size})."

        return (
            self.data_buffer_list_[rank][storage_offset : storage_offset + ask_size]
            .view(dtype)
            .reshape(sizes)
        )

    def get_signal(self, rank: int, *sizes: Any, storage_offset: int = 0):
        # force signal buffer use uint32
        dtype = torch.uint32

        assert 0 <= rank <= self.world_size

        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            sizes = tuple(sizes[0])
        else:
            sizes = tuple(sizes)

        numel = list(accumulate(sizes, func=mul))[-1]
        ask_size = numel * dtype.itemsize

        assert (
            storage_offset + ask_size <= self.signal_size
        ), f"The requested buffer size(got {storage_offset} + {ask_size} = {storage_offset + ask_size}) exceeds the size of the hold buffer(got {self.signal_size})."

        return (
            self.signal_buffer_list_[rank][storage_offset : storage_offset + ask_size]
            .view(dtype)
            .reshape(sizes)
        )

    def get_multimem_buff(
        self, *sizes: Any, dtype: Optional[torch.dtype] = None, storage_offset: int = 0
    ) -> torch.Tensor:
        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            sizes = tuple(sizes[0])
        else:
            sizes = tuple(sizes)

        if dtype is None:
            dtype = torch.get_default_dtype()

        numel = list(accumulate(sizes, func=mul))[-1]
        ask_size = numel * dtype.itemsize

        assert (
            storage_offset + ask_size <= self.buffer_size
        ), f"The requested buffer size(got {storage_offset} + {ask_size} = {storage_offset + ask_size}) exceeds the size of the hold buffer(got {self.buffer_size})."

        return (
            self.multimem_data_buffer_[storage_offset : storage_offset + ask_size]
            .view(dtype)
            .reshape(sizes)
        )

    def get_multimem_signal(self, *sizes: Any, storage_offset: int = 0) -> torch.Tensor:
        # force signal buffer use uint32
        dtype = torch.uint32

        if len(sizes) == 1 and isinstance(sizes[0], Sequence):
            sizes = tuple(sizes[0])
        else:
            sizes = tuple(sizes)

        numel = list(accumulate(sizes, func=mul))[-1]
        ask_size = numel * dtype.itemsize

        assert (
            storage_offset + ask_size <= self.signal_size
        ), f"The requested buffer size(got {storage_offset} + {ask_size} = {storage_offset + ask_size}) exceeds the size of the hold buffer(got {self.signal_size})."

        return (
            self.multimem_signal_buffer_[storage_offset : storage_offset + ask_size]
            .view(dtype)
            .reshape(sizes)
        )

    def barrier(self, channel: int = 0, timeout_ms: int = 0):
        # call a singal kernel for gpu barrier
        pass
