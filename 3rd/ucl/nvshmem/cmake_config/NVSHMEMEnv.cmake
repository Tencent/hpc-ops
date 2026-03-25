# This file includes boiler plate code for transforming
# environment variables to CMake variables.
# RULES:
# 1. Options only need an entry here if they are on by default
# 2. Paths only need an entry here if they need a non-empty default

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

macro(nvshmem_add_default_on_option OPTION_NAME DESCRIPTION)
  if (DEFINED ENV{${OPTION_NAME}})
    set(${OPTION_NAME}_DEFAULT $ENV{${OPTION_NAME}})
  else()
    set(${OPTION_NAME}_DEFAULT ON)
  endif()
  option(${OPTION_NAME} ${DESCRIPTION} ${${OPTION_NAME}_DEFAULT})
  message( "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()

macro (nvshmem_add_default_off_option OPTION_NAME DESCRIPTION)
  option(${OPTION_NAME} ${DESCRIPTION} $ENV{${OPTION_NAME}})
  message( "${OPTION_NAME}: ${${OPTION_NAME}}")
endmacro()



macro (nvshmem_add_default_environment_path PATH_NAME DEFAULT_VALUE DESCRIPTION)
if (DEFINED ENV{${PATH_NAME}})
    set(${PATH_NAME}_DEFAULT $ENV{${PATH_NAME}})
  else()
    set(${PATH_NAME}_DEFAULT ${DEFAULT_VALUE})
  endif()
  set(${PATH_NAME} ${${PATH_NAME}_DEFAULT} CACHE PATH ${DESCRIPTION})
  message( "${PATH_NAME}: ${${PATH_NAME}}")
endmacro()

message( "                                   **NVSHMEM OPTIONS OVERVIEW**                                   \n")
message( "Options set to ON by Default")
message( "___________________________________________________________________________________________________")
message( "\n__BOOTSTRAP__\n")
nvshmem_add_default_on_option(NVSHMEM_MPI_SUPPORT "Enable compilation of the MPI bootstrap and MPI-specific code")

message( "\n__TRANSPORT__\n")
nvshmem_add_default_on_option(NVSHMEM_IBRC_SUPPORT "Enable compilation of the IBRC remote transport")

message( "\n__FUNCTIONALITY__\n")
nvshmem_add_default_on_option(NVSHMEM_USE_GDRCOPY "Enable compilation of GDRCopy offload paths for atomics in remote transports")
nvshmem_add_default_on_option(NVSHMEM_USE_MLX5DV "Enable directNIC feature")
nvshmem_add_default_on_option(NVSHMEM_NVTX "Enable NVSHMEM NVTX support")

message( "\n__BUILD__\n")
nvshmem_add_default_on_option(NVSHMEM_BUILD_TESTS "Build tests")
nvshmem_add_default_on_option(NVSHMEM_BUILD_EXAMPLES "Build examples")
nvshmem_add_default_off_option(NVSHMEM_BUILD_CUTLASS_EXAMPLES "Build CUTLASS examples")
nvshmem_add_default_on_option(NVSHMEM_BUILD_PYTHON_LIB "Builds NVSHMEM4PY Wheels and Python source tarballs")
message( "___________________________________________________________________________________________________\n\n")

message( "Options set to OFF by Default")
message( "___________________________________________________________________________________________________")
message( "\n__BOOTSTRAP__\n")
nvshmem_add_default_off_option(NVSHMEM_DEFAULT_PMI2 "Sets PMI2 as the default bootstrap")
nvshmem_add_default_off_option(NVSHMEM_DEFAULT_PMIX "Sets PMIX as the default bootstrap")
nvshmem_add_default_off_option(NVSHMEM_PMIX_SUPPORT "Enable Compilation of the PMIX bootstrap and PMIX specific code")
nvshmem_add_default_off_option(NVSHMEM_SHMEM_SUPPORT "Enable Compilation of the SHMEM bootstrap and SHMEM specific code")
nvshmem_add_default_on_option(NVSHMEM_PMI_SUPPORT "Enable PMI bootstrap")
nvshmem_add_default_on_option(NVSHMEM_PMI2_SUPPORT "Enable PMI2 bootstrap")

message( "\n__TRANSPORT__\n")
nvshmem_add_default_off_option(NVSHMEM_DEFAULT_UCX "Sets UCX as the default remote transport")
nvshmem_add_default_off_option(NVSHMEM_IBGDA_SUPPORT "Enable compilation of the IBGDA remote transport")
nvshmem_add_default_off_option(NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY "Force the IBGDA remote transport to only use GPUMEM.")
nvshmem_add_default_off_option(NVSHMEM_IBDEVX_SUPPORT "Enable compilation of the IBDevX remote transport")
nvshmem_add_default_off_option(NVSHMEM_LIBFABRIC_SUPPORT "Enable compilation of the libfabric remote transport")
nvshmem_add_default_off_option(NVSHMEM_UCX_SUPPORT "Enable compilation of the UCX remote transport")

message( "\n__FUNCTIONALITY__\n")
nvshmem_add_default_off_option(NVSHMEM_USE_DLMALLOC "Set dlmalloc as the NVSHMEM heap allocation method")
nvshmem_add_default_off_option(NVSHMEM_USE_NCCL "Enable compilation of NVSHMEM NCCL dependent code")
nvshmem_add_default_off_option(NVSHMEM_ENABLE_ALL_DEVICE_INLINING "Inline all device library code")
nvshmem_add_default_off_option(NVSHMEM_GPU_COLL_USE_LDST "Enables Load/Store in NVSHMEM collective operations")
nvshmem_add_default_off_option(NVSHMEM_TIMEOUT_DEVICE_POLLING "Enable timeouts for NVSHMEM device-side polling functions (e.g. wait_until)")

message( "\n__BUILD__\n")
nvshmem_add_default_off_option(NVSHMEM_DEBUG "Toggles NVSHMEM debug compilation settings")
nvshmem_add_default_off_option(NVSHMEM_DEVEL "Toggles NVSHMEM devel compilation settings")
nvshmem_add_default_off_option(NVSHMEM_TRACE "Enable NVSHMEM trace print events")
nvshmem_add_default_off_option(NVSHMEM_VERBOSE "Enable the ptxas verbose compilation option")
nvshmem_add_default_off_option(NVSHMEM_ENV_ALL "Display all runtime environment variables, regardless of build settings")
message( "___________________________________________________________________________________________________\n\n")


message( "Paths of dependencies")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_environment_path(CUDA_HOME "/usr/local/cuda" "path to CUDA installation")
nvshmem_add_default_environment_path(GDRCOPY_HOME "/usr/local/gdrdrv" "path to GDRCOPY installation")
nvshmem_add_default_environment_path(LIBFABRIC_HOME "/usr/local/libfabric" "path to libfabric installation")
nvshmem_add_default_environment_path(NVSHMEM_CLANG_DIR "" "path to force cmake to look for clang when compiling the bitcode library.")
nvshmem_add_default_environment_path(MPI_HOME "/usr/local/ompi" "path to MPI installation")
nvshmem_add_default_environment_path(NCCL_HOME "/usr/local/nccl" "path to NCCL installation")
nvshmem_add_default_environment_path(NVSHMEM_PREFIX "/usr/local/nvshmem" "path to NVSHMEM install directory.")
nvshmem_add_default_environment_path(PMIX_HOME "/usr" "path to PMIX installation")
nvshmem_add_default_environment_path(SHMEM_HOME "${MPI_HOME}" "path to SHMEM installation")
nvshmem_add_default_environment_path(UCX_HOME "/usr/local/ucx" "path to UCX installation")
message( "___________________________________________________________________________________________________\n\n")


message( "Packaging specific options set to OFF by default")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_off_option(NVSHMEM_BUILD_PACKAGES "Build package dependencies - gates all other packaging variables")
nvshmem_add_default_off_option(NVSHMEM_BUILD_RPM_PACKAGE "Build RPM package")
nvshmem_add_default_off_option(NVSHMEM_BUILD_DEB_PACKAGE "Build DEB package")
nvshmem_add_default_off_option(NVSHMEM_BUILD_TGZ_PACKAGE "Build TGZ package")
if (NOT NVSHMEM_BUILD_PACKAGES)
  nvshmem_add_default_off_option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library")
endif()
message( "___________________________________________________________________________________________________\n\n")

message( "Packaging specific options set to ON by default")
message( "___________________________________________________________________________________________________")
nvshmem_add_default_on_option(NVSHMEM_BUILD_HYDRA_LAUNCHER "Enables Building the hydra launcher package")
nvshmem_add_default_on_option(NVSHMEM_BUILD_TXZ_PACKAGE "Build TXZ package")
if (NVSHMEM_BUILD_PACKAGES)
  nvshmem_add_default_on_option(NVSHMEM_BUILD_BITCODE_LIBRARY "Build the nvshmem_device bitcode library")
endif()
message( "___________________________________________________________________________________________________\n\n")


### HIDDEN ###
if (DEFINED ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{NVSHMEM_DEVICELIB_CUDA_HOME})
elseif(DEFINED ENV{CUDA_HOME})
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT $ENV{CUDA_HOME})
else()
  set(NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT "/usr/local/cuda")
endif()
set(NVSHMEM_DEVICELIB_CUDA_HOME ${NVSHMEM_DEVICELIB_CUDA_HOME_DEFAULT} CACHE PATH "path to CUDA installation")

option(NVSHMEM_INSTALL_FUNCTIONAL_TESTS "Install functional tests" $ENV{NVSHMEM_INSTALL_FUNCTIONAL_TESTS})
option(NVSHMEM_TEST_STATIC_LIB "Force tests to link only against the combined nvshmem.a binary" $ENV{NVSHMEM_TEST_STATIC_LIB})

# Set a variable for the nvshmem build information in NVSHMEM
# TODO actually fill this variable with useful information.
set(INFO_BUILD_VARS
  "\"NVSHMEM_DEBUG=${NVSHMEM_DEBUG} \
NVSHMEM_DEVEL=${NVSHMEM_DEVEL} \
NVSHMEM_DEFAULT_PMI2=${NVSHMEM_DEFAULT_PMI2} \
NVSHMEM_DEFAULT_PMIX=${NVSHMEM_DEFAULT_PMIX} \
NVSHMEM_DEFAULT_UCX=${NVSHMEM_DEFAULT_UCX} \
NVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING} \
NVSHMEM_GPU_COLL_USE_LDST=${NVSHMEM_GPU_COLL_USE_LDST} \
NVSHMEM_IBGDA_SUPPORT=${NVSHMEM_IBGDA_SUPPORT} \
NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=${NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY} \
NVSHMEM_IBDEVX_SUPPORT=${NVSHMEM_IBDEVX_SUPPORT} \
NVSHMEM_IBRC_SUPPORT=${NVSHMEM_IBRC_SUPPORT} \
NVSHMEM_LIBFABRIC_SUPPORT=${NVSHMEM_LIBFABRIC_SUPPORT} \
NVSHMEM_MPI_SUPPORT=${NVSHMEM_MPI_SUPPORT} \
NVSHMEM_NVTX=${NVSHMEM_NVTX} \
NVSHMEM_PMIX_SUPPORT=${NVSHMEM_PMIX_SUPPORT} \
NVSHMEM_PMI_SUPPORT=${NVSHMEM_PMI_SUPPORT} \
NVSHMEM_PMI2_SUPPORT=${NVSHMEM_PMI2_SUPPORT} \
NVSHMEM_SHMEM_SUPPORT=${NVSHMEM_SHMEM_SUPPORT} \
NVSHMEM_TEST_STATIC_LIB=${NVSHMEM_TEST_STATIC_LIB} \
NVSHMEM_TIMEOUT_DEVICE_POLLING=${NVSHMEM_TIMEOUT_DEVICE_POLLING} \
NVSHMEM_TRACE=${NVSHMEM_TRACE} \
NVSHMEM_UCX_SUPPORT=${NVSHMEM_UCX_SUPPORT} \
NVSHMEM_USE_DLMALLOC=${NVSHMEM_USE_DLMALLOC} \
NVSHMEM_USE_NCCL=${NVSHMEM_USE_NCCL} \
NVSHMEM_USE_GDRCOPY=${NVSHMEM_USE_GDRCOPY} \
NVSHMEM_VERBOSE=${NVSHMEM_VERBOSE} \
CUDA_HOME=${CUDA_HOME} \
GDRCOPY_HOME=${GDRCOPY_HOME} \
LIBFABRIC_HOME=${LIBFABRIC_HOME} \
MPI_HOME=${MPI_HOME} \
NCCL_HOME=${NCCL_HOME} \
NVSHMEM_PREFIX=${NVSHMEM_PREFIX} \
PMIX_HOME=${PMIX_HOME} \
SHMEM_HOME=${SHMEM_HOME} \
UCX_HOME=${UCX_HOME}\"")
