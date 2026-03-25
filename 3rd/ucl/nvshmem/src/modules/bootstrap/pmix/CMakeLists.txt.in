set(SOURCE_LIST bootstrap_pmix.c)

nvshmem_add_bootstrap(nvshmem_bootstrap_pmix ${SOURCE_LIST})

find_library(PMIX_lib NAMES pmix HINTS "${PMIX_HOME}/lib")

target_link_libraries(nvshmem_bootstrap_pmix PRIVATE ${PMIX_lib})
target_include_directories(nvshmem_bootstrap_pmix PRIVATE ${PMIX_HOME}/include)
