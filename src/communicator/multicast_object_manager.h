// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTICAST_OBJECT_MANAGER_H_
#define SRC_COMMUNICATOR_MULTICAST_OBJECT_MANAGER_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace hpc {
namespace communicator {

class MulticastObjectManager {
 public:
  MulticastObjectManager(int device_id, int num_devices);
  ~MulticastObjectManager();

  bool CreateMulticastObjAndExportFd(int *fd, int64_t bytes);
  bool CreateMulticastObjByImportFd(int fd);

  bool AllocateMemoryAndBindToMulticastObj(void **multi_ptr, void **local_ptr);

 private:
  int device_id_;
  int num_devices_;
  int64_t bytes_;
  int64_t aligned_bytes_;
  CUmemGenericAllocationHandle multi_handle_;
  CUmemGenericAllocationHandle local_handle_;
};

}  // namespace communicator
}  // namespace hpc

#endif  // SRC_COMMUNICATOR_MULTICAST_OBJECT_MANAGER_H_
