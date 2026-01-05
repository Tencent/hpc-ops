# this requires cmake 3.19
include(CheckCompilerFlag)
include(CheckSourceCompiles)
include(ExternalProject)

check_compiler_flag(CUDA -t4 NVCC_THREADS)

set(BOOTSTRAP_VERSION_MAJOR 3)
set(BOOTSTRAP_VERSION_MINOR 0)
set(BOOTSTRAP_VERSION_PATCH 0)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  add_compile_options(-Wno-vla-cxx-extension)
endif()

if(NOT CMAKE_CUDA_ARCHITECTURES_UNDEFINED)
    list(APPEND my_cmake_cache_args
        -DCMAKE_CUDA_ARCHITECTURES:STRING=${CMAKE_CUDA_ARCHITECTURES}
    )
endif()

ExternalProject_Add(nvshmem_device_project
                    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device
                    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/device
                    CMAKE_ARGS  -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                                -DNVSHMEM_PREFIX=${NVSHMEM_PREFIX}
                                -DNVSHMEM_DEBUG=${NVSHMEM_DEBUG}
                                -DNVSHMEM_DEVICELIB_CUDA_HOME=${NVSHMEM_DEVICELIB_CUDA_HOME}
                                -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING}
                                -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
                                -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
                                -DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}
                                -DPROJECT_VERSION=${PROJECT_VERSION}
				$ENV{CMAKE_CCACHE_FLAGS}
                    CMAKE_CACHE_ARGS ${my_cmake_cache_args}
                    STEP_TARGETS build install
                    BUILD_ALWAYS TRUE
                    BUILD_BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a
                    )

add_library(nvshmem_device STATIC IMPORTED)
set_property(TARGET nvshmem_device PROPERTY IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a)

## Start bitcode library configurations

if(CUDAToolkit_VERSION_MAJOR LESS 13)
  set(LIB_CXX_STANDARD 11)
else()
  set(LIB_CXX_STANDARD 17)
endif()

if(NVSHMEM_BUILD_BITCODE_LIBRARY)
  if(NOT NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
    configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in llvm_lib/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
  endif()
  if(CUDAToolkit_VERSION_MAJOR LESS 12)
    set(BITCODE_LIB_ARCH "sm_70")
  else()
    set(BITCODE_LIB_ARCH "sm_90")
  endif()

  if(CUDAToolkit_VERSION_MAJOR LESS 13)
    set(BITCODE_CXX_STD "c++11")
  else()
    set(BITCODE_CXX_STD "c++17")
  endif()

  # Note - today we do optimization to avoid having a too large bitcode file. There are, as of clang 18, some existing
  # limitations around compiling device libraries which make optimizing this library fine. Once these issues are fixed
  # we will have to remove optimizations to enable more flexible builds.
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.bc
                    COMMAND clang -c -emit-llvm -O1 -std=${BITCODE_CXX_STD} -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only
                    --cuda-gpu-arch=${BITCODE_LIB_ARCH} -I ${CMAKE_CURRENT_SOURCE_DIR}/include -I ${CUDA_HOME}/include/cccl
                    -I ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib
                    -D__clang_llvm_bitcode_lib__ ${CMAKE_CURRENT_SOURCE_DIR}/device/init/init_device.cu -o libnvshmem_device.bc.unoptimized
                    COMMAND opt --passes='internalize,inline,globaldce'
                    -internalize-public-api-list='nvshmem_*,nvshmemx_*,nvshmemi_device_lib_version_d,nvshmemi_device_state_d'
                    libnvshmem_device.bc.unoptimized -o libnvshmem_device.bc
                    COMMAND llvm-dis libnvshmem_device.bc
                    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/bitcode_lib_cleanup.sh ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib/libnvshmem_device.ll ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib/libnvshmem_device.ll.new
                    COMMAND llvm-as libnvshmem_device.ll.new -o ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.bc
                    COMMAND rm libnvshmem_device.bc.unoptimized libnvshmem_device.ll libnvshmem_device.ll.new libnvshmem_device.bc
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/llvm_lib
                    DEPENDS device/init/init_device.cu)

  add_custom_target(libnvshmem_device_bitcode ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.bc)
  install(
          FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.bc
          DESTINATION lib/
          PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                      GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
  )
endif()

## End bitcode library configurations

if(NVSHMEM_BUILD_EXAMPLES)
set(NVSHMEM_EXAMPLES_RELEASE_PREFIX "bin/examples")
  add_subdirectory(${CMAKE_SOURCE_DIR}/examples ${CMAKE_SOURCE_DIR}/build/examples)
endif()
if(NVSHMEM_BUILD_TESTS)
  set(NVSHMEM_PERFTEST_RELEASE_PREFIX "bin/perftest")
  add_subdirectory(${CMAKE_SOURCE_DIR}/perftest ${CMAKE_SOURCE_DIR}/build/perftest)
  if(EXISTS ${CMAKE_SOURCE_DIR}/test)
    if (NVSHMEM_INSTALL_FUNCTIONAL_TESTS)
      set(NVSHMEM_TEST_RELEASE_PREFIX "bin/test")
    endif()
    add_subdirectory(${CMAKE_SOURCE_DIR}/test ${CMAKE_SOURCE_DIR}/build/test)
  endif()
endif()

# Start bootstrap definitions
if(NVSHMEM_PMI_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmi SHARED
  )
  set(ACTIVE_BOOTSTRAPS nvshmem_bootstrap_pmi)
  target_sources(nvshmem_bootstrap_pmi PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                               modules/bootstrap/pmi/bootstrap_pmi.cpp
                                               modules/bootstrap/pmi/simple-pmi/simple_pmi.cpp
                                               modules/bootstrap/pmi/simple-pmi/simple_pmiutil.cpp)
  target_include_directories(nvshmem_bootstrap_pmi
                             PRIVATE include
                             modules/bootstrap/common
                             modules/bootstrap/pmi/simple-pmi
  )
endif()

if(NVSHMEM_PMI2_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmi2 SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmi2)
  target_sources(nvshmem_bootstrap_pmi2 PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                                modules/bootstrap/pmi/bootstrap_pmi.cpp
                                                modules/bootstrap/pmi/pmi-2/pmi2_api.c
                                                modules/bootstrap/pmi/pmi-2/pmi2_util.c)
  target_compile_definitions(nvshmem_bootstrap_pmi2 PRIVATE NVSHMEM_BUILD_PMI2)
  target_link_options(nvshmem_bootstrap_pmi2 PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_bootstrap.sym")
  target_include_directories(nvshmem_bootstrap_pmi2
                             PRIVATE include
                             modules/bootstrap/common
                             modules/bootstrap/pmi/pmi-2
  )
else ()
set(NVSHMEM_DEFAULT_PMI2 OFF)
endif()

if(NVSHMEM_PMIX_SUPPORT)
  add_library(
    nvshmem_bootstrap_pmix SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_pmix)
  target_sources(nvshmem_bootstrap_pmix PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                                modules/bootstrap/pmix/bootstrap_pmix.c)
  find_library(PMIX_lib NAMES pmix HINTS "${PMIX_HOME}/lib")
  target_link_libraries(nvshmem_bootstrap_pmix PRIVATE ${PMIX_lib})
  target_include_directories(nvshmem_bootstrap_pmix
                             PRIVATE include
                             modules/bootstrap/common
                             ${PMIX_HOME}/include

  )
else()
set(NVSHMEM_DEFAULT_PMIX OFF)
endif()

if(NVSHMEM_MPI_SUPPORT)
  add_library(
    nvshmem_bootstrap_mpi SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_mpi)
  set_target_properties(nvshmem_bootstrap_mpi PROPERTIES C_STANDARD 11)
  target_sources(nvshmem_bootstrap_mpi PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                               modules/bootstrap/mpi/bootstrap_mpi.c)
  target_link_libraries(nvshmem_bootstrap_mpi PRIVATE MPI::MPI_C)
  target_include_directories(nvshmem_bootstrap_mpi
                             PRIVATE include
                             modules/bootstrap/common
                             include/modules/common
  )
endif()

if(NVSHMEM_SHMEM_SUPPORT)
  add_library(
    nvshmem_bootstrap_shmem SHARED
  )
  set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_shmem)
  set_target_properties(nvshmem_bootstrap_shmem PROPERTIES C_STANDARD 11)
  target_sources(nvshmem_bootstrap_shmem PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                                 modules/bootstrap/shmem/bootstrap_shmem.c)
  target_link_libraries(nvshmem_bootstrap_shmem PRIVATE shmem)
  target_include_directories(nvshmem_bootstrap_shmem
                             PRIVATE include
                             modules/bootstrap/common
                             include/modules/common
  )
endif()

# Always compile uid bootstrap as no dependency on external sources or libraries
add_library(nvshmem_bootstrap_uid SHARED)
set(ACTIVE_BOOTSTRAPS ${ACTIVE_BOOTSTRAPS} nvshmem_bootstrap_uid)
set_target_properties(nvshmem_bootstrap_uid
                      PROPERTIES C_STANDARD 11
                      CXX_STANDARD_REQUIRED ON
                      CXX_STANDARD 11)
target_sources(nvshmem_bootstrap_uid PRIVATE modules/bootstrap/common/bootstrap_util.cpp
                                             modules/bootstrap/uid/ncclSocket/ncclsocket_socket.cpp
					                                   modules/bootstrap/uid/bootstrap_uid.cpp)
target_include_directories(nvshmem_bootstrap_uid
		                       PRIVATE include
                           include/bootstrap_device_host
                           include/internal/bootstrap_host
                           include/internal/bootstrap_host_transport
                           modules/bootstrap/common
			                     modules/bootstrap/uid/ncclSocket
)


set_target_properties(${ACTIVE_BOOTSTRAPS}
                      PROPERTIES PREFIX ""
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${BOOTSTRAP_VERSION_MAJOR}.${BOOTSTRAP_VERSION_MINOR}.${BOOTSTRAP_VERSION_PATCH}
                      SOVERSION ${BOOTSTRAP_VERSION_MAJOR})

# End bootstrap definitions

# Start nvshmem lib definitions
add_library(
  nvshmem STATIC
)

add_library(
  nvshmem_host SHARED
)

## Start generic variable configuration
set_target_properties(nvshmem nvshmem_host
                      PROPERTIES POSITION_INDEPENDENT_CODE ON
                      CXX_STANDARD_REQUIRED ON
                      CUDA_STANDARD_REQUIRED ON
                      CXX_STANDARD "${LIB_CXX_STANDARD}"
                      CUDA_STANDARD "${LIB_CXX_STANDARD}"
                      CUDA_SEPARABLE_COMPILATION ON
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
		      VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
		      SOVERSION ${PROJECT_VERSION_MAJOR}
)

macro(nvshmem_library_set_base_config LIBNAME)
  target_compile_definitions(${LIBNAME}
    PRIVATE $<$<CONFIG:Debug>:_NVSHMEM_DEBUG;NVSHMEM_IBGDA_DEBUG>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","x86_64">,NVSHMEM_X86_64,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","ppc64le">,__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS;NVSHMEM_PPC64LE,>
    $<IF:$<STREQUAL:"${CMAKE_HOST_SYSTEM_PROCESSOR}","aarch64">,NVSHMEM_AARCH64,>
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:__STDC_LIMIT_MACROS;__STDC_CONSTANT_MACROS>
  )

  target_compile_options(${LIBNAME}
    INTERFACE $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVSHMEM_VERBOSE}>>:-Xptxas -v>
    PRIVATE $<IF:$<CONFIG:Debug>,-O0;-g;,-O3>
    $<$<AND:$<BOOL:${NVSHMEM_VERBOSE}>,$<COMPILE_LANGUAGE:CUDA>>:-Xptxas -v>
    $<IF:$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>,-O0;-g;-G;,-O3>
    $<IF:$<STREQUAL:${CMAKE_HOST_SYSTEM_PROCESSOR},"x86_64">,-msse,>
    $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<BOOL:${NVCC_THREADS}>>:-t4>
  )

  if(NVSHMEM_DEVEL)
    target_compile_options(
      ${LIBNAME}
      PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Werror
              all-warnings>
              $<$<COMPILE_LANGUAGE:CXX>:-Werror
              -Wall
              -Wextra
              -Wno-unused-function
              -Wno-unused-parameter
              -Wno-missing-field-initializers>
    )
  endif()
endmacro()

set(NVSHMEM_PYTHON_SOURCES
    ${CMAKE_SOURCE_DIR}/nvshmem4py/nvshmem/
)

set(NVSHMEM_DEVICE_SOURCES
    device/init/init_device.cu
    device/launch/collective_launch.cpp
)

set(NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT
    host/comm/rma.cu
    host/stream/comm/quiet_on_stream.cu
    host/stream/comm/cuda_interface_sync.cu
    host/stream/coll/alltoall/alltoall.cu
    host/stream/coll/barrier/barrier.cu
    host/stream/coll/broadcast/broadcast.cu
    host/stream/coll/fcollect/fcollect.cu
    host/stream/coll/rdxn/reduce_and.cu
    host/stream/coll/rdxn/reduce_or.cu
    host/stream/coll/rdxn/reduce_xor.cu
    host/stream/coll/rdxn/reduce_min.cu
    host/stream/coll/rdxn/reduce_max.cu
    host/stream/coll/rdxn/reduce_prod.cu
    host/stream/coll/rdxn/reduce_sum.cu
    host/stream/coll/rdxn/reduce_team.cu
    host/stream/coll/reducescatter/reducescatter_and.cu
    host/stream/coll/reducescatter/reducescatter_or.cu
    host/stream/coll/reducescatter/reducescatter_xor.cu
    host/stream/coll/reducescatter/reducescatter_min.cu
    host/stream/coll/reducescatter/reducescatter_max.cu
    host/stream/coll/reducescatter/reducescatter_prod.cu
    host/stream/coll/reducescatter/reducescatter_sum.cu
)

set(NVSHMEM_HOST_SOURCES
    host/bootstrap/bootstrap.cpp
    host/bootstrap/bootstrap_loader.cpp
    host/coll/cpu_coll.cpp
    host/coll/alltoall/alltoall.cpp
    host/coll/alltoall/alltoall_on_stream.cpp
    host/coll/barrier/barrier.cpp
    host/coll/barrier/barrier_on_stream.cpp
    host/coll/broadcast/broadcast.cpp
    host/coll/broadcast/broadcast_on_stream.cpp
    host/coll/fcollect/fcollect.cpp
    host/coll/fcollect/fcollect_on_stream.cpp
    host/coll/rdxn/rdxn.cpp
    host/coll/rdxn/rdxn_on_stream.cpp
    host/coll/reducescatter/reducescatter.cpp
    host/coll/reducescatter/reducescatter_on_stream.cpp
    host/comm/putget.cpp
    host/comm/fence.cpp
    host/comm/quiet.cpp
    host/comm/sync.cpp
    host/comm/amo.cpp
    host/proxy/proxy.cpp
    host/transport/transport.cpp
    host/transport/p2p/p2p.cpp
    host/init/cudawrap.cpp
    host/init/init.cu
    host/init/init_nvtx.cpp
    host/init/nvmlwrap.cpp
    host/init/query_host.cpp
    host/mem/mem.cpp
    host/mem/mem_heap.cpp
    host/mem/mem_transport.cpp
    host/team/team.cu
    host/team/team_internal.cpp
    host/team/team_internal_cuda.cu
    host/team/team_internal_nvls.cpp
    host/topo/topo.cpp
    host/util/cs.cpp
    host/util/debug.cpp
    host/util/env_vars.cpp
    host/util/util.cpp
    host/util/sockets.cpp
    host/util/shared_memory.cpp
)

target_include_directories(
  nvshmem
  PRIVATE include
          include/host/env
          host/stream/coll
          host/coll
          host/topo
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

target_include_directories(
  nvshmem_host
  PRIVATE include
          include/host/env
          device/coll
          host/coll
          host/stream/coll
          host/topo
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
            $<INSTALL_INTERFACE:include>
)

target_link_libraries(nvshmem PRIVATE CUDA::cudart_static)
target_link_libraries(nvshmem_host PRIVATE CUDA::cudart_static)
if (CCCL_FOUND AND CUDAToolkit_VERSION_MAJOR GREATER 12)
  target_link_libraries(nvshmem PUBLIC CCCL::CCCL)
  target_link_libraries(nvshmem_host PUBLIC CCCL::CCCL)
endif()
nvshmem_library_set_base_config(nvshmem)
nvshmem_library_set_base_config(nvshmem_host)
## End generic variable configuration

## Start transports
set(TRANSPORT_VERSION_MAJOR 3)
set(TRANSPORT_VERSION_MINOR 0)
set(TRANSPORT_VERSION_PATCH 0)

check_source_compiles(C
  "#include <infiniband/verbs.h>
  int main(void) { int x = IBV_ACCESS_RELAXED_ORDERING; return 1; }"
  HAVE_IBV_ACCESS_RELAXED_ORDERING
)

set(ACTIVE_TRANSPORTS)

if(NVSHMEM_UCX_SUPPORT)
  add_library(
    nvshmem_transport_ucx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ucx)
  target_sources(nvshmem_transport_ucx PRIVATE
                 modules/transport/ucx/ucx.cpp
                 modules/transport/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_ucx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ucx
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ucx PRIVATE ucx::ucs ucx::ucp CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ucx PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ucx PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ucx)
endif()

if(NVSHMEM_IBRC_SUPPORT)
  add_library(
    nvshmem_transport_ibrc SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibrc)
  target_sources(nvshmem_transport_ibrc PRIVATE
                 modules/transport/ibrc/ibrc.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
  )
  target_link_options(nvshmem_transport_ibrc PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibrc
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ibrc PRIVATE CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ibrc PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ibrc PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibrc PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibrc)
endif()

if(NVSHMEM_IBDEVX_SUPPORT OR NVSHMEM_IBGDA_SUPPORT)
  find_library(MLX5_lib NAMES mlx5)
endif()

if(NVSHMEM_IBDEVX_SUPPORT)
  add_library(
    nvshmem_transport_ibdevx SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibdevx)
  target_sources(nvshmem_transport_ibdevx PRIVATE
                 modules/transport/ibdevx/ibdevx.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibdevx PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibdevx
                             PRIVATE include
                             modules/transport/common
  )
  target_link_libraries(nvshmem_transport_ibdevx PRIVATE ${MLX5_lib} CUDA::cudart_static)
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibdevx PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibdevx)
endif()

if(NVSHMEM_IBGDA_SUPPORT)
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED; return 1; }"
    HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED
  )
  check_source_compiles(C
    "#include <infiniband/mlx5dv.h>
    int main(void) { int x = MLX5DV_UMEM_MASK_DMABUF; return 1; }"
    HAVE_MLX5DV_UMEM_MASK_DMABUF
  )
  add_library(
    nvshmem_transport_ibgda SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_ibgda)
  target_sources(nvshmem_transport_ibgda PRIVATE
                 modules/transport/ibgda/ibgda.cpp
                 modules/transport/common/transport_common.cpp
                 modules/transport/common/transport_ib_common.cpp
                 modules/transport/common/transport_mlx5_common.cpp
  )
  target_link_options(nvshmem_transport_ibgda PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_ibgda
                             PRIVATE include
                             modules/transport/common

  )
  target_link_libraries(nvshmem_transport_ibgda PRIVATE ${MLX5_lib} CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_ibgda PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_ibgda PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  if(HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED=${HAVE_MLX5DV_UAR_ALLOC_TYPE_NC_DEDICATED})
  endif()
  if(HAVE_MLX5DV_UMEM_MASK_DMABUF)
      target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_MLX5DV_UMEM_MASK_DMABUF=${HAVE_MLX5DV_UMEM_MASK_DMABUF})
  endif()
  if(HAVE_IBV_ACCESS_RELAXED_ORDERING)
    target_compile_definitions(nvshmem_transport_ibgda PRIVATE HAVE_IBV_ACCESS_RELAXED_ORDERING=${HAVE_IBV_ACCESS_RELAXED_ORDERING})
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_ibgda)
endif()

if(NVSHMEM_LIBFABRIC_SUPPORT)
  find_library(FABRIC_lib NAMES fabric HINTS "${LIBFABRIC_HOME}/lib" "${LIBFABRIC_HOME}/lib64")
  add_library(
    nvshmem_transport_libfabric SHARED
  )
  nvshmem_library_set_base_config(nvshmem_transport_libfabric)
  target_sources(nvshmem_transport_libfabric PRIVATE
                 modules/transport/libfabric/libfabric.cpp
                 modules/transport/common/transport_common.cpp
  )
  target_link_options(nvshmem_transport_libfabric PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_transport.sym")
  target_include_directories(nvshmem_transport_libfabric
                             PRIVATE include
                             modules/transport/common
                             "${LIBFABRIC_HOME}/include/"
  )
  target_link_libraries(nvshmem_transport_libfabric PRIVATE ${FABRIC_lib} CUDA::cudart_static)
  if(NVSHMEM_USE_GDRCOPY)
    target_include_directories(nvshmem_transport_libfabric PRIVATE ${GDRCOPY_INCLUDE})
    target_sources(nvshmem_transport_libfabric PRIVATE
                   modules/transport/common/transport_gdr_common.cpp
    )
  endif()
  set(ACTIVE_TRANSPORTS ${ACTIVE_TRANSPORTS} nvshmem_transport_libfabric)
endif()

set_target_properties(${ACTIVE_TRANSPORTS}
                      PROPERTIES PREFIX ""
                      CXX_STANDARD_REQUIRED ON
                      CXX_STANDARD 11
                      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
                      VERSION ${TRANSPORT_VERSION_MAJOR}.${TRANSPORT_VERSION_MINOR}.${TRANSPORT_VERSION_PATCH}
                      SOVERSION ${TRANSPORT_VERSION_MAJOR})
## End transports

## Start dependencies
if(NVSHMEM_USE_NCCL)
  target_include_directories(nvshmem PRIVATE ${NCCL_INCLUDE})
  target_include_directories(nvshmem_host PRIVATE ${NCCL_INCLUDE})
endif()

if(NVSHMEM_USE_DLMALLOC)
  set(NVSHMEM_HOST_SOURCES
      ${NVSHMEM_HOST_SOURCES}
      host/mem/dlmalloc.cpp
  )
else()
set(NVSHMEM_HOST_SOURCES
    ${NVSHMEM_HOST_SOURCES}
    host/mem/custom_malloc.cpp
  )
endif()
## End dependecies

## Start Numba support requirements
set(INC_EXPORTS
    ${INC_EXPORTS}
    non_abi/device/pt-to-pt/transfer_device.cuh
)
configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/device/pt-to-pt/transfer_device.cuh COPYONLY)
## End Numba support requirements

## Start inlining requirements
if(NOT NVSHMEM_ENABLE_ALL_DEVICE_INLINING)
set(INC_EXPORTS
    ${INC_EXPORTS}
    non_abi/device/pt-to-pt/nvshmemi_transfer_api.cuh
)
set(NVSHMEM_DEVICE_SOURCES
    ${NVSHMEM_DEVICE_SOURCES}
    device/comm/transfer_device.cu
)
configure_file(include/non_abi/device/pt-to-pt/transfer_device.cuh.in ${CMAKE_CURRENT_SOURCE_DIR}/device/comm/transfer_device.cu COPYONLY)
endif()
## End inlining requirements

## Start final lib prep
target_link_libraries(nvshmem_host PRIVATE nvshmem_device)
target_link_options(nvshmem_host PRIVATE "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../nvshmem_host.sym")
set_source_files_properties(${NVSHMEM_HOST_SOURCES} ${NVSHMEM_DEVICE_SOURCES} PROPERTIES COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=32>)
target_sources(nvshmem PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT} ${NVSHMEM_DEVICE_SOURCES} ${NVSHMEM_DEVICE_SOURCES_NOMAXREGCOUNT})
target_sources(nvshmem_host PRIVATE ${NVSHMEM_HOST_SOURCES} ${NVSHMEM_HOST_SOURCES_NOMAXREGCOUNT})

add_library(NVSHMEM::nvshmem ALIAS nvshmem)

## End final lib prep
# End nvshmem lib definitions

# Start nvshmem-info definitions.
add_executable(nvshmem-info)
target_sources(nvshmem-info PRIVATE bin/nvshmem-info.cpp)
set_target_properties(nvshmem-info PROPERTIES CXX_STANDARD_REQUIRED ON CXX_STANDARD ${LIB_CXX_STANDARD} RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
target_link_libraries(nvshmem-info CUDA::cudart_static CUDA::cuda_driver nvshmem)
target_include_directories(nvshmem-info PRIVATE include include/host/env)
# End nvshmem-info definitions.

# Start header configuration and export
configure_file(include/non_abi/nvshmem_version.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_version.h @ONLY)
configure_file(include/non_abi/nvshmem_build_options.h.in ${CMAKE_CURRENT_SOURCE_DIR}/include/non_abi/nvshmem_build_options.h)

set(INC_EXPORTS
    ${INC_EXPORTS}
    bootstrap_device_host/nvshmem_uniqueid.h
    device_host/nvshmem_proxy_channel.h
    device_host/nvshmem_types.h
    device_host_transport/nvshmem_common_transport.h
    non_abi/device/common/nvshmemi_common_device.cuh
    non_abi/device/threadgroup/nvshmemi_common_device_defines.cuh
    device/nvshmem_coll_defines.cuh
    device/nvshmem_defines.h
    device/nvshmem_device_macros.h
    device/nvshmemx_collective_launch_apis.h
    device/nvshmemx_coll_defines.cuh
    device/nvshmemx_defines.h
    device/tile/nvshmemx_tile_api.hpp
    device/tile/nvshmemx_tile_api_defines.cuh
    non_abi/device/coll/alltoall.cuh
    non_abi/device/coll/barrier.cuh
    non_abi/device/coll/broadcast.cuh
    non_abi/device/coll/defines.cuh
    non_abi/device/coll/fcollect.cuh
    non_abi/device/coll/reduce.cuh
    non_abi/device/coll/reducescatter.cuh
    non_abi/device/coll/utils.cuh
    non_abi/device/wait/nvshmemi_wait_until_apis.cuh
    non_abi/device/pt-to-pt/ibgda_device.cuh
    non_abi/device/pt-to-pt/proxy_device.cuh
    non_abi/device/pt-to-pt/utils_device.h
    non_abi/device/common/nvshmemi_tile_utils.cuh
    device_host/nvshmem_common.cuh
    device_host/nvshmem_tensor.h
    device_host_transport/nvshmem_common_ibgda.h
    device_host_transport/nvshmem_constants.h
    non_abi/device/team/nvshmemi_team_defines.cuh
    host/nvshmem_api.h
    host/nvshmem_coll_api.h
    host/nvshmem_macros.h
    host/nvshmemx_api.h
    host/nvshmemx_coll_api.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
    non_abi/nvshmem_version.h
    nvshmem.h
    nvshmem_host.h
    nvshmemx.h
)

foreach(INC ${INC_EXPORTS})
  configure_file(include/${INC} ${CMAKE_CURRENT_BINARY_DIR}/include/${INC} COPYONLY)
endforeach(INC ${INC_EXPORTS})
# End header configuration and export

# start share configuration

configure_file(${CMAKE_SOURCE_DIR}/nvshmem_bootstrap.sym
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/nvshmem_bootstrap.sym COPYONLY)

set(BOOTSTRAP_EXPORTS
    common/bootstrap_util.cpp
    common/bootstrap_util.h
    common/env_defs.h
    mpi/bootstrap_mpi.c
    pmi/bootstrap_pmi.cpp
    pmix/bootstrap_pmix.c
    shmem/bootstrap_shmem.c
    uid/bootstrap_uid.cpp
    uid/bootstrap_uid_remap.h
    uid/ncclSocket/ncclsocket_checks.h
    uid/ncclSocket/ncclsocket_debug.h
    uid/ncclSocket/ncclsocket_nccl.h
    uid/ncclSocket/ncclsocket_param.h
    uid/ncclSocket/ncclsocket_socket.hpp
    uid/ncclSocket/ncclsocket_socket.cpp
    uid/ncclSocket/ncclsocket_utils.h
    uid/bootstrap_uid_types.hpp
)

foreach(SHARE ${BOOTSTRAP_EXPORTS})
  get_filename_component(SHAREFILE ${SHARE} NAME)
  configure_file(modules/bootstrap/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${BOOTSTRAP_EXPORTS})

set(BOOTSTRAP_INC_EXPORTS
    bootstrap_device_host/nvshmem_uniqueid.h
    bootstrap_host_transport/env_defs_internal.h
    internal/bootstrap_host/nvshmemi_bootstrap.h
    internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h
    non_abi/nvshmem_version.h
    non_abi/nvshmemx_error.h
)

foreach(SHARE ${BOOTSTRAP_INC_EXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/include/${SHARE} COPYONLY)
endforeach(SHARE ${BOOTSTRAP_INC_EXPORTS})

set(BOOTSTRAP_CMAKE_EXPORTS
    CMakeLists.txt
    common/CMakeLists.txt
    mpi/CMakeLists.txt
    pmi/CMakeLists.txt
    pmix/CMakeLists.txt
    shmem/CMakeLists.txt
    uid/CMakeLists.txt
)

foreach(SHARE ${BOOTSTRAP_CMAKE_EXPORTS})
  configure_file(modules/bootstrap/${SHARE}.in ${CMAKE_CURRENT_BINARY_DIR}/share/src/bootstrap-plugins/${SHARE} @ONLY)
endforeach(SHARE ${BOOTSTRAP_CMAKE_EXPORTS})

configure_file(${CMAKE_SOURCE_DIR}/nvshmem_transport.sym
               ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/nvshmem_transport.sym COPYONLY)

set(TRANSPORT_INC_EXPORTS
    bootstrap_host_transport/env_defs_internal.h
    device_host_transport/nvshmem_common_ibgda.h
    device_host_transport/nvshmem_common_transport.h
    device_host_transport/nvshmem_constants.h
    internal/bootstrap_host_transport/nvshmemi_bootstrap_defines.h
    internal/host_transport/cudawrap.h
    internal/host_transport/nvshmemi_transport_defines.h
    internal/host_transport/transport.h
    non_abi/nvshmem_version.h
    non_abi/nvshmemx_error.h
    non_abi/nvshmem_build_options.h
)

foreach(SHARE ${TRANSPORT_INC_EXPORTS})
  configure_file(include/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/include/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_INC_EXPORTS})

set(TRANSPORT_EXPORTS
    common/env_defs.h
    common/mlx5_ifc.h
    common/mlx5_prm.h
    common/transport_common.cpp
    common/transport_common.h
    common/transport_gdr_common.cpp
    common/transport_gdr_common.h
    common/transport_ib_common.cpp
    common/transport_ib_common.h
    common/transport_mlx5_common.cpp
    common/transport_mlx5_common.h
    ibdevx/ibdevx.cpp
    ibdevx/ibdevx.h
    ibgda/ibgda.cpp
    ibrc/ibrc.cpp
    libfabric/libfabric.cpp
    libfabric/libfabric.h
    ucx/ucx.cpp
    ucx/ucx.h
)

foreach(SHARE ${TRANSPORT_EXPORTS})
  configure_file(modules/transport/${SHARE} ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} COPYONLY)
endforeach(SHARE ${TRANSPORT_EXPORTS})

set(TRANSPORT_CMAKE_EXPORTS
    CMakeLists.txt
    common/CMakeLists.txt
    libfabric/CMakeLists.txt
    ibdevx/CMakeLists.txt
    ibgda/CMakeLists.txt
    ibrc/CMakeLists.txt
    ucx/CMakeLists.txt
)

foreach(SHARE ${TRANSPORT_CMAKE_EXPORTS})
  configure_file(modules/transport/${SHARE}.in ${CMAKE_CURRENT_BINARY_DIR}/share/src/transport-plugins/${SHARE} @ONLY)
endforeach(SHARE ${TRANSPORT_CMAKE_EXPORTS})
# end share configuration

# start custom builds for nvshmem packages
add_custom_target(git_commit ALL COMMAND test -f git_commit.txt || git rev-parse HEAD > git_commit.txt || echo "not built from a git repo" > git_commit.txt
                  COMMAND grep -q NVSHMEM_MAJOR git_commit.txt || echo "NVSHMEM_MAJOR := ${PROJECT_VERSION_MAJOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_MINOR git_commit.txt || echo "NVSHMEM_MINOR := ${PROJECT_VERSION_MINOR}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PATCH git_commit.txt || echo "NVSHMEM_PATCH := ${PROJECT_VERSION_PATCH}" >> git_commit.txt
                  COMMAND grep -q NVSHMEM_PACKAGE git_commit.txt || echo "NVSHMEM_PACKAGE := ${PROJECT_VERSION_TWEAK}" >> git_commit.txt
                  COMMAND cp git_commit.txt version.txt
                  BYPRODUCTS ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
                             ${CMAKE_CURRENT_SOURCE_DIR}/../version.txt
                  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(NVSHMEM_BUILD_PACKAGES AND NVSHMEM_BUILD_HYDRA_LAUNCHER)
  add_custom_target(hydra
                    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../scripts/install_hydra.sh ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build ${CMAKE_CURRENT_BINARY_DIR}
                    COMMAND rm -rf ${CMAKE_CURRENT_SOURCE_DIR}/hydra_build
                    BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_nameserver
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_persist
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/hydra_pmi_proxy
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun
                               ${CMAKE_CURRENT_BINARY_DIR}/bin/nvshmrun.hydra
                  )
  add_dependencies(nvshmem git_commit hydra)
endif()
add_dependencies(nvshmem_host nvshmem_device_project-build)

# Start Installation definitions
install(CODE "
  # Patch NVSHMEM target files
  file(GLOB_RECURSE targets_files \"${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/Export/*/NVSHMEM*Targets*.cmake\")
  foreach(targets_file IN LISTS targets_files)
    message(STATUS \"Processing targets_file=\${targets_file}\")
    file(READ \"\${targets_file}\" _file_contents)
    string(REPLACE \"\\\${_IMPORT_PREFIX}/bin\" \"\\\${NVSHMEM_BIN_DIR}\" _file_contents \"\${_file_contents}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/include\" \"\\\${NVSHMEM_INCLUDE_DIR}\" _file_contents \"\${_file_contents}\")
    string(REPLACE \"\\\${_IMPORT_PREFIX}/lib\" \"\\\${NVSHMEM_LIB_DIR}\" _file_contents \"\${_file_contents}\")
    file(WRITE \"\${targets_file}\" \"\${_file_contents}\")
  endforeach()
")


# This makes sure the device config file is included as well
install(CODE "
  # Patch NVSHMEM target files
  file(GLOB_RECURSE targets_files \"${CMAKE_BINARY_DIR}/lib/cmake/NVSHMEM*Targets*.cmake\")
  foreach(targets_file IN LISTS targets_files)
    message(STATUS \"Processing targets_file=\${targets_file}\")
    file(READ \"\${targets_file}\" _file_contents)
    string(REPLACE \"\\\${_IMPORT_PREFIX}/bin\" \"\\\${NVSHMEM_BIN_DIR}\" _file_contents \"\${_file_contents}\")
        string(REPLACE \"\\\${_IMPORT_PREFIX}/include\" \"\\\${NVSHMEM_INCLUDE_DIR}\" _file_contents \"\${_file_contents}\")
    string(REPLACE \"\\\${_IMPORT_PREFIX}/lib\" \"\\\${NVSHMEM_LIB_DIR}\" _file_contents \"\${_file_contents}\")
    file(WRITE \"\${targets_file}\" \"\${_file_contents}\")
  endforeach()
")

install(
  EXPORT NVSHMEMTargets
  NAMESPACE nvshmem::
  DESTINATION lib/cmake/nvshmem/
  COMPONENT Export
)

# ${CMAKE_CURRENT_BINARY_DIR}/NVSHMEMConfig.cmake
install(FILES ${CMAKE_BINARY_DIR}/NVSHMEMVersion.cmake
        DESTINATION lib/cmake/nvshmem
)

install(FILES ${CMAKE_BINARY_DIR}/NVSHMEMConfig.cmake
        DESTINATION lib/cmake/nvshmem
	)

export(
  EXPORT NVSHMEMTargets
  NAMESPACE nvshmem::
  FILE "NVSHMEMTargets.cmake"
)

install(
	TARGETS nvshmem nvshmem_host nvshmem-info ${ACTIVE_BOOTSTRAPS} ${ACTIVE_TRANSPORTS}
  EXPORT NVSHMEMTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES
  DESTINATION include
  PUBLIC_HEADER DESTINATION include
)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/lib/libnvshmem_device.a
  DESTINATION lib/
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
              GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(
  FILES ${CMAKE_BINARY_DIR}/lib/cmake/nvshmem/NVSHMEMDeviceTargets.cmake
  DESTINATION lib/cmake/nvshmem/
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ WORLD_READ
)

install(
  FILES ${CMAKE_BINARY_DIR}/lib/cmake/nvshmem/NVSHMEMDeviceTargets-$<LOWER_CASE:${CMAKE_BUILD_TYPE}>.cmake
  DESTINATION lib/cmake/nvshmem/
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ WORLD_READ
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib/
  DESTINATION lib
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin/
  DESTINATION bin
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                   GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)
if(NVSHMEM_BUILD_TESTS)
  install(
    DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/share/src/
    DESTINATION share/src/
  )
endif()
install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/
  DESTINATION include/
  PATTERN "modules" EXCLUDE
)
if(NVSHMEM_BUILD_EXAMPLES)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../examples/
    DESTINATION share/src/examples/
    PATTERN "examples_install" EXCLUDE
  )
endif()
if(NVSHMEM_BUILD_TESTS)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../perftest/
    DESTINATION share/src/perftest/
    PATTERN "perftest_install" EXCLUDE
    PATTERN "perftestCommon.py" EXCLUDE
    PATTERN "perftest-ib.list" EXCLUDE
    PATTERN "perftest-p2p-nvlink.list" EXCLUDE
    PATTERN "perftest-p2p-pcie.list" EXCLUDE
    PATTERN "perftestRunner.py" EXCLUDE
    PATTERN "perfTestRunnerSlurm.py" EXCLUDE
)
endif()

if(NVSHMEM_BUILD_TESTS)
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../changelog
    DESTINATION ./
  )
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../License.txt
    DESTINATION ./
  )
  install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../version.txt
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/../git_commit.txt
    DESTINATION ./
  )
endif()

include(CMakePackageConfigHelpers)

if(NVSHMEM_BUILD_TGZ_PACKAGE)
  list(APPEND CPACK_GENERATOR "TGZ")
endif()

if(NVSHMEM_BUILD_TXZ_PACKAGE)
  list(APPEND CPACK_GENERATOR "TXZ")
endif()

if(NVSHMEM_BUILD_RPM_PACKAGE)
  list(APPEND CPACK_GENERATOR "RPM")
endif()

if(NVSHMEM_BUILD_DEB_PACKAGE)
  list(APPEND CPACK_GENERATOR "DEB")
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "x86_64")
  set(PACKAGE_SUFFIX "x86_64")
elseif(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "ppc64le")
  set(PACKAGE_SUFFIX "ppc64le")
else()
  set(PACKAGE_SUFFIX "aarch64")
endif()

set(CPACK_SOURCE_INSTALLED_DIRECTORIES "${CMAKE_SOURCE_DIR}/;/")


set(CPACK_SOURCE_IGNORE_FILES
"/install/;\
/build/;\
/.vscode;\
/include_gdrcopy;\
/include_nccl;\
/.clang-format;\
/.gitignore;\
/.git/;\
/.gitlab-ci.yml;\
/bind.sh;\
/README;\
/testcommon.py;\
/examples/examples_install/;\
/include/non_abi/nvshmem_build_options.h$;\
/include/non_abi/nvshmem_version.h$;\
/nvshmem4py/cmake/addCybind.cmake;\
/nvshmem4py/cybind_assets/;\
/perftest/perftest_install/;\
/test/test_install/;
/scripts/create_imex.sh;\
/scripts/dependency_installer.sh;\
/scripts/include_checker.sh;\
/scripts/iwyu_c_cpp_check.sh;\
/scripts/iwyu_header_cuda_check.sh;\
/scripts/iwyu_output_parser.py;\
/scripts/iwyu_test_functionality.sh;\
/scripts/testrunner.pl;\
/src/build/;")

if(NOT NVSHMEM_INSTALL_FUNCTIONAL_TESTS)
  set(CPACK_SOURCE_IGNORE_FILES ${CPACK_SOURCE_IGNORE_FILES}"/test/;/docker/;/nvshmem4py/test/;")
endif()


set(CPACK_PACKAGE_NAME "libnvshmem")
set(CPACK_PACKAGE_VENDOR "NVIDIA")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://developer.nvidia.com/nvshmem")
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../pkg/nvshmem_package_description.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "NVSHMEM - An implementation of OpenSHMEM for NVIDIA GPUs.")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_FILE_NAME "libnvshmem_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}+cuda${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}_${PACKAGE_SUFFIX}")
set(CPACK_SOURCE_PACKAGE_FILE_NAME "nvshmem_src_${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}-${PROJECT_VERSION_TWEAK}")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../License.txt")
set(CPACK_PACKAGE_CONTACT "nvshmem@nvidia.com")

# RPM-Specific Configurations
set(CPACK_RPM_PACKAGE_GROUP "Development/Libraries")
set(CPACK_RPM_PACKAGE_LICENSE "LICENSE AGREEMENT FOR NVIDIA SOFTWARE DEVELOPMENT KITS")
set(CPACK_RPM_PACKAGE_AUTOREQ 0)
set(CPACK_RPM_PACKAGE_REQUIRES "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PRE "/sbin/ldconfig")
set(CPACK_RPM_PACKAGE_REQUIRES_PREUN "/sbin/ldconfig")

include(CPack)

add_custom_target(package_src_target
    COMMAND "${CMAKE_COMMAND}" --build . --target package_source
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    COMMENT "Package Build"
    DEPENDS build_bindings_cybind 
    VERBATIM
)
# End Installation definitions


