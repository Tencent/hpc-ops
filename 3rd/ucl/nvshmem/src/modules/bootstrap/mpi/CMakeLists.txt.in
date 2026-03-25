set(SOURCE_LIST bootstrap_mpi.c)

nvshmem_add_bootstrap(nvshmem_bootstrap_mpi ${SOURCE_LIST})

find_package(MPI REQUIRED)

target_link_libraries(nvshmem_bootstrap_mpi PRIVATE MPI::MPI_C)
