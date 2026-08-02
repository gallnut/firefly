set(CPM_DOWNLOAD_VERSION 0.40.0)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
set(CPM_DOWNLOAD_TEMP "${CPM_DOWNLOAD_LOCATION}.tmp")

if(NOT DEFINED ENV{CPM_SOURCE_CACHE})
    set(ENV{CPM_SOURCE_CACHE} "${CMAKE_BINARY_DIR}/_deps")
endif()

set(CPM_DOWNLOAD_SIZE 0)
if(EXISTS "${CPM_DOWNLOAD_LOCATION}")
    file(SIZE "${CPM_DOWNLOAD_LOCATION}" CPM_DOWNLOAD_SIZE)
endif()

if(CPM_DOWNLOAD_SIZE EQUAL 0)
    message(STATUS "Downloading CPM.cmake...")
    file(REMOVE "${CPM_DOWNLOAD_TEMP}")
    file(DOWNLOAD
         https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
         "${CPM_DOWNLOAD_TEMP}"
         STATUS CPM_DOWNLOAD_STATUS
    )
    list(GET CPM_DOWNLOAD_STATUS 0 CPM_DOWNLOAD_ERROR)
    if(CPM_DOWNLOAD_ERROR)
        file(REMOVE "${CPM_DOWNLOAD_TEMP}")
        message(FATAL_ERROR "Failed to download CPM.cmake. Set CPM_SOURCE_CACHE or provide network access: ${CPM_DOWNLOAD_STATUS}")
    endif()
    file(RENAME "${CPM_DOWNLOAD_TEMP}" "${CPM_DOWNLOAD_LOCATION}")
endif()

include("${CPM_DOWNLOAD_LOCATION}")
