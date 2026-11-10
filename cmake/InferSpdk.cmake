if(TARGET InferSpdk::dependencies)
    return()
endif()

find_package(PkgConfig REQUIRED)
set(INFER_SPDK_ROOT "" CACHE PATH "SPDK source/build tree or installed SDK prefix")
set(INFER_SPDK_MODULES
    spdk_event spdk_event_bdev spdk_event_iobuf spdk_bdev_nvme
    spdk_bdev_malloc spdk_env_dpdk spdk_syslibs)
if(INFER_SPDK_URING)
    list(APPEND INFER_SPDK_MODULES spdk_bdev_uring liburing>=2.3)
endif()
set(INFER_SPDK_SAVED_PKG_CONFIG_PATH "$ENV{PKG_CONFIG_PATH}")
if(INFER_SPDK_ROOT)
    set(ENV{PKG_CONFIG_PATH}
        "${INFER_SPDK_ROOT}/build/lib/pkgconfig:${INFER_SPDK_ROOT}/lib/pkgconfig:${INFER_SPDK_ROOT}/lib64/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    set(INFER_SPDK_FOUND FALSE)
endif()
pkg_check_modules(INFER_SPDK REQUIRED ${INFER_SPDK_MODULES})
set(ENV{PKG_CONFIG_PATH} "${INFER_SPDK_SAVED_PKG_CONFIG_PATH}")

# Resolve SDK paths in each consumer, rather than exporting build-machine paths.
add_library(InferSpdk::dependencies INTERFACE IMPORTED GLOBAL)
set_target_properties(InferSpdk::dependencies PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${INFER_SPDK_INCLUDE_DIRS}"
    INTERFACE_LINK_DIRECTORIES "${INFER_SPDK_LIBRARY_DIRS}"
    INTERFACE_COMPILE_OPTIONS "${INFER_SPDK_CFLAGS_OTHER}"
    INTERFACE_LINK_OPTIONS "-Wl,--no-undefined"
    INTERFACE_LINK_LIBRARIES
        "-Wl,--no-as-needed;-Wl,--start-group;-Wl,--whole-archive;${INFER_SPDK_STATIC_LDFLAGS};-Wl,--no-whole-archive;-Wl,--end-group;-Wl,--as-needed")
