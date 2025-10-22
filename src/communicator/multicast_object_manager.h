// Copyright 2025 hpc-ops authors

#ifndef SRC_COMMUNICATOR_MULTICAST_OBJECT_MANAGER_H_
#define SRC_COMMUNICATOR_MULTICAST_OBJECT_MANAGER_H_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdint.h>

#include <memory>
#include <tuple>

namespace hpc {
namespace communicator {

class MulticastObjectManager {
 public:
  MulticastObjectManager(int device_id, int num_devices);
  ~MulticastObjectManager();

  bool CreateMemoryObjAndExportFd(int *fd, int64_t bytes, std::shared_ptr<void> *obj, int *device);
  bool CreateMemoryObjByImportFd(int fd, int64_t bytes, std::shared_ptr<void> *obj, int *device);

  bool CreateMulticastObjAndExportFd(int *fd, int64_t bytes, std::shared_ptr<void> *handle);
  bool CreateMulticastObjByImportFd(int fd, int64_t bytes, std::shared_ptr<void> *handle);
  bool MapHandleToAddresableObj(std::shared_ptr<void> multi_handle,
                                std::shared_ptr<void> *multi_obj, int *device, int64_t bytes);

  bool BindLocalMemoryObjToMulticastObj(std::shared_ptr<void> obj, int device,
                                        std::shared_ptr<void> multi_obj, int multi_device,
                                        int64_t bytes);

 private:
  static constexpr int64_t kAlignment = 2 * 1024 * 1024;  // 2MB

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
