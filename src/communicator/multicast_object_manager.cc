// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_object_manager.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "src/communicator/type.h"

#define CUCHECK(cmd)                                                \
  do {                                                              \
    auto r = cmd;                                                   \
    if (r != CUDA_SUCCESS) {                                        \
      const char *err = nullptr;                                    \
      cuGetErrorString(r, &err);                                    \
      std::string error(#cmd);                                      \
      throw std::runtime_error(error + " fail, with err = " + err); \
    }                                                               \
  } while (0)

namespace hpc {
namespace communicator {

static int64_t align_to(int64_t bytes, int64_t alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

MulticastObjectManager::MulticastObjectManager(int device_id, int num_devices) {
  multi_handle_ = 0;
  local_handle_ = 0;

  device_id_ = device_id;
  num_devices_ = num_devices;
}

MulticastObjectManager::~MulticastObjectManager() {
  device_id_ = -1;
  num_devices_ = -1;
}

bool MulticastObjectManager::CreateMulticastObjAndExportFd(int *fd, int64_t bytes) {
  cudaSetDevice(device_id_);

  int64_t aligned_bytes = align_to(bytes, kAlignment);

  bytes_ = bytes;
  aligned_bytes_ = aligned_bytes;

  // create multicast
  CUmulticastObjectProp prop = {};
  prop.flags = 0;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.numDevices = num_devices_;
  prop.size = aligned_bytes;

  CUCHECK(cuMulticastCreate(&multi_handle_, &prop));

  // export file descriptor
  CUCHECK(
      cuMemExportToShareableHandle(fd, multi_handle_, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

  // add device to multicast
  CUCHECK(cuMulticastAddDevice(multi_handle_, device_id_));

  return true;
}

bool MulticastObjectManager::CreateMulticastObjByImportFd(int fd, int64_t bytes) {
  cudaSetDevice(device_id_);

  int64_t aligned_bytes = align_to(bytes, kAlignment);
  bytes_ = bytes;
  aligned_bytes_ = aligned_bytes;

  int64_t fd64 = fd;
  // work rank, import multicast
  CUCHECK(cuMemImportFromShareableHandle(&multi_handle_, reinterpret_cast<void *>(fd64),
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

  CUmemAllocationProp prop;
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, multi_handle_));
  CUCHECK(cuMulticastAddDevice(multi_handle_, device_id_));

  return true;
}

MulticastTensors MulticastObjectManager::AllocateMemoryAndBindToMulticastObj() {
  int curr_device_id;
  cudaGetDevice(&curr_device_id);
  cudaSetDevice(device_id_);

  CUdeviceptr local_ptr;
  {
    CUmemAllocationProp prop = {};
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_id_;
    CUCHECK(cuMemCreate(&local_handle_, aligned_bytes_, &prop, 0));

    CUCHECK(cuMemAddressReserve(&local_ptr, aligned_bytes_, 2 * 1024 * 1024, 0, 0));
    CUCHECK(cuMemMap(local_ptr, aligned_bytes_, 0, local_handle_, 0));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device_id_;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(local_ptr, aligned_bytes_, &desc, 1));
  }

  CUdeviceptr multi_ptr;
  {
    CUCHECK(cuMemAddressReserve(&multi_ptr, aligned_bytes_, 2 * 1024 * 1024, 0, 0));
    CUCHECK(cuMemMap(multi_ptr, aligned_bytes_, 0, multi_handle_, 0));

    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device_id_;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUCHECK(cuMemSetAccess(multi_ptr, aligned_bytes_, &desc, 1));
  }

  CUCHECK(cuMulticastBindMem(multi_handle_, 0, local_handle_, 0, aligned_bytes_, 0));

  int multi_device = -1;
  {
    CUmemAllocationProp prop = {};
    CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, multi_handle_));
    multi_device = prop.location.id;
  }

  auto local_deleter = [handle = local_handle_, size = aligned_bytes_](void *ptr) {
    CUdeviceptr dptr = (CUdeviceptr)ptr;
    CUCHECK(cuMemUnmap(dptr, size));
    CUCHECK(cuMemAddressFree(dptr, size));
    CUCHECK(cuMemRelease(handle));
  };

  std::shared_ptr<void> local_sobj(reinterpret_cast<void *>(local_ptr), local_deleter);

  auto multi_deleter = [handle = multi_handle_, size = aligned_bytes_, dev = device_id_,
                        local_sobj](void *ptr) {
    CUdeviceptr dptr = (CUdeviceptr)ptr;

    CUCHECK(cuMulticastUnbind(handle, dev, 0, size));
    CUCHECK(cuMemUnmap(dptr, size));
    CUCHECK(cuMemAddressFree(dptr, size));
    CUCHECK(cuMemRelease(handle));
  };

  std::shared_ptr<void> multi_sobj(reinterpret_cast<void *>(multi_ptr), multi_deleter);

  cudaSetDevice(curr_device_id);

  MulticastTensors tensors{multi_sobj, local_sobj, multi_device, device_id_, true};

  return tensors;
}

}  // namespace communicator
}  // namespace hpc
