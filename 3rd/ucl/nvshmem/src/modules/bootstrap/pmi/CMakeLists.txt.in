set(SOURCE_LIST bootstrap_pmi.cpp)

if(NVSHMEM_BUILD_PMI_BOOTSTRAP)
  nvshmem_add_bootstrap(nvshmem_bootstrap_pmi ${SOURCE_LIST})

  find_library(PMI_lib NAMES pmi HINTS "${PMI_HOME}/lib")

  target_link_libraries(nvshmem_bootstrap_pmi PRIVATE ${PMI_lib})
  target_compile_definitions(nvshmem_bootstrap_pmi PRIVATE NVSHMEM_CUSTOM_BOOTSTRAP_BUILD)
  target_include_directories(nvshmem_bootstrap_pmi PRIVATE ${PMI_HOME}/include)
endif()

if(NVSHMEM_BUILD_PMI2_BOOTSTRAP)
  nvshmem_add_bootstrap(nvshmem_bootstrap_pmi2 ${SOURCE_LIST})

  find_library(PMI2_lib NAMES pmi2 HINTS "${PM2I_HOME}/lib")

  target_compile_definitions(nvshmem_bootstrap_pmi2 PRIVATE NVSHMEM_BUILD_PMI2)
  target_link_libraries(nvshmem_bootstrap_pmi2 PRIVATE ${PMI2_lib})
  target_include_directories(nvshmem_bootstrap_pmi2 PRIVATE ${PMI2_HOME}/include)
endif()
