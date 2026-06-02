// Copyright 2025 hpc-ops authors

#include "src/communicator/multicast_object_manager.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

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

#define CUDACHECK(cmd)                                                                  \
  do {                                                                                  \
    auto r = cmd;                                                                       \
    if (r != cudaSuccess) {                                                             \
      auto err = cudaGetLastError();                                                    \
      std::string error(#cmd);                                                          \
      throw std::runtime_error(error + " fail, with err = " + cudaGetErrorString(err)); \
    }                                                                                   \
  } while (0)

namespace hpc {
namespace communicator {

static int64_t align_to(int64_t bytes, int64_t alignment) {
  return ((bytes + alignment - 1) / alignment) * alignment;
}

static std::shared_ptr<void> ReserveAddrMapHandleAndSetAccess(CUmemGenericAllocationHandle handle,
                                                              int64_t aligned_bytes, int *device) {
  CUmemAllocationProp prop;
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, handle));

  CUdeviceptr ptr;
  CUCHECK(cuMemAddressReserve(&ptr, aligned_bytes, 2 * 1024 * 1024, 0, 0));
  CUCHECK(cuMemMap(ptr, aligned_bytes, 0, handle, 0));

  CUmemAccessDesc desc = {};
  desc.location.type = prop.location.type;
  desc.location.id = *device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess(ptr, aligned_bytes, &desc, 1));

  auto deleter = [handle = handle, size = aligned_bytes](void *ptr) {
    CUdeviceptr dptr = (CUdeviceptr)ptr;
    CUCHECK(cuMemUnmap(dptr, size));
    CUCHECK(cuMemAddressFree(dptr, size));
    CUCHECK(cuMemRelease(handle));
  };

  std::shared_ptr<void> sobj(reinterpret_cast<void *>(ptr), deleter);
  *device = prop.location.id;

  return sobj;
}

static std::shared_ptr<void> ReserveAddrMapHandleAndSetAccessMulticast(
    CUmemGenericAllocationHandle handle, int64_t aligned_bytes, int *device) {
  CUmemAllocationProp prop;
  CUCHECK(cuMemGetAllocationPropertiesFromHandle(&prop, handle));

  CUdeviceptr ptr;
  CUCHECK(cuMemAddressReserve(&ptr, aligned_bytes, 2 * 1024 * 1024, 0, 0));
  CUCHECK(cuMemMap(ptr, aligned_bytes, 0, handle, 0));

  CUmemAccessDesc desc = {};
  desc.location.type = prop.location.type;
  desc.location.id = *device;
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CUCHECK(cuMemSetAccess(ptr, aligned_bytes, &desc, 1));

  auto deleter = [handle = handle, dev = prop.location.id, size = aligned_bytes](void *ptr) {
    CUdeviceptr dptr = (CUdeviceptr)ptr;
    CUCHECK(cuMemUnmap(dptr, size));
    CUCHECK(cuMemAddressFree(dptr, size));
    CUCHECK(cuMemRelease(handle));
  };

  std::shared_ptr<void> sobj(reinterpret_cast<void *>(ptr), deleter);
  *device = prop.location.id;

  return sobj;
}

MulticastObjectManager::MulticastObjectManager(int device_id, int num_devices) {
  device_id_ = device_id;
  num_devices_ = num_devices;

  // check device count and init cuda context
  int num_device_actual = -1;
  CUDACHECK(cudaGetDeviceCount(&num_device_actual));
  if (num_device_actual < num_devices_) {
    std::stringstream ss;
    ss << "available gpu device count(" << num_device_actual
       << ") is less than communicator world_size(" << num_devices << ")";
    throw std::runtime_error(ss.str());
  }
}

MulticastObjectManager::~MulticastObjectManager() {
  device_id_ = -1;
  num_devices_ = -1;
}

bool MulticastObjectManager::CreateMemoryObjAndExportFd(int *fd, int64_t bytes,
                                                        std::shared_ptr<void> *obj, int *device) {
  int64_t aligned_bytes = align_to(bytes, kAlignment);

  CUmemGenericAllocationHandle handle;
  CUmemAllocationProp prop = {};
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id_;
  CUCHECK(cuMemCreate(&handle, aligned_bytes, &prop, 0));

  CUCHECK(cuMemExportToShareableHandle(fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

  *device = device_id_;
  *obj = ReserveAddrMapHandleAndSetAccess(handle, aligned_bytes, device);
  return true;
}

bool MulticastObjectManager::CreateMemoryObjByImportFd(int fd, int64_t bytes,
                                                       std::shared_ptr<void> *obj, int *device) {
  int64_t aligned_bytes = align_to(bytes, kAlignment);

  CUmemGenericAllocationHandle handle;
  int64_t fd64 = fd;
  // work rank, import multicast
  CUCHECK(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void *>(fd64),
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

  *device = device_id_;
  *obj = ReserveAddrMapHandleAndSetAccess(handle, aligned_bytes, device);
  return true;
}

// root rank, create multicast obj and export it
bool MulticastObjectManager::CreateMulticastHandleAndExportFd(int *fd, int64_t bytes,
                                                              std::shared_ptr<void> *multi_handle) {
  int64_t aligned_bytes = align_to(bytes, kAlignment);

  // create multicast
  CUmulticastObjectProp prop = {};
  prop.flags = 0;
  prop.handleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  prop.numDevices = num_devices_;
  prop.size = aligned_bytes;

  CUmemGenericAllocationHandle handle;
  CUCHECK(cuMulticastCreate(&handle, &prop));
  // export file descriptor
  CUCHECK(cuMemExportToShareableHandle(fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

  // add device to multicast
  CUCHECK(cuMulticastAddDevice(handle, device_id_));

  *multi_handle = std::make_shared<CUmemGenericAllocationHandle>(handle);

  return true;
}

// non-root rank, import multicast
bool MulticastObjectManager::CreateMulticastHandleByImportFd(int fd, int64_t bytes,
                                                             std::shared_ptr<void> *multi_handle) {
  int64_t fd64 = fd;
  CUmemGenericAllocationHandle handle;
  CUCHECK(cuMemImportFromShareableHandle(&handle, reinterpret_cast<void *>(fd64),
                                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

  // add device to multicast
  CUCHECK(cuMulticastAddDevice(handle, device_id_));

  *multi_handle = std::make_shared<CUmemGenericAllocationHandle>(handle);

  return true;
}

bool MulticastObjectManager::MapHandleToMulticastObj(std::shared_ptr<void> multi_handle,
                                                     std::shared_ptr<void> *multi_obj, int *device,
                                                     int64_t bytes) {
  int64_t aligned_bytes = align_to(bytes, kAlignment);

  auto shandle = std::static_pointer_cast<CUmemGenericAllocationHandle>(multi_handle);

  *device = device_id_;
  *multi_obj = ReserveAddrMapHandleAndSetAccessMulticast(*shandle, aligned_bytes, device);
  return true;
}

bool MulticastObjectManager::BindLocalMemoryObjToMulticastObj(std::shared_ptr<void> local_obj,
                                                              int device,
                                                              std::shared_ptr<void> multi_obj,
                                                              int multi_device, int64_t bytes) {
  int64_t aligned_bytes = align_to(bytes, kAlignment);

  CUmemGenericAllocationHandle multi_handle;
  CUCHECK(cuMemRetainAllocationHandle(&multi_handle, multi_obj.get()));

  CUmemGenericAllocationHandle local_handle;
  CUCHECK(cuMemRetainAllocationHandle(&local_handle, local_obj.get()));

  // bind local memory to multicast
  CUCHECK(cuMulticastBindMem(multi_handle, 0, local_handle, 0, aligned_bytes, 0));

  return true;
}

}  // namespace communicator
}  // namespace hpc
