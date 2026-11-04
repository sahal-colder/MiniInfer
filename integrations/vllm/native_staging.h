#pragma once

#include "infer/storage/ssd_device.h"
#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace miniinfer_staging {
// DLPack's stable C layout, kept local so the storage bridge does not depend
// on Torch headers or link its runtime merely to hand over an owned CPU view.
struct Device {
    std::int32_t type, id;
};
struct DataType {
    std::uint8_t code, bits;
    std::uint16_t lanes;
};
struct Tensor {
    void* data;
    Device device;
    std::int32_t ndim;
    DataType dtype;
    std::int64_t* shape;
    std::int64_t* strides;
    std::uint64_t byte_offset;
};
struct ManagedTensor {
    Tensor dl_tensor;
    void* manager_ctx;
    void (*deleter)(ManagedTensor*);
};
inline void require_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}
struct DeviceScope {
    int previous = 0;
    explicit DeviceScope(int device) {
        require_cuda(cudaGetDevice(&previous), "staging get device");
        require_cuda(cudaSetDevice(device), "staging set device");
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
struct TensorOwner {
    ManagedTensor managed{};
    std::shared_ptr<infer::SSDDmaBuffer> buffer;
    std::int64_t shape = 0;
    int device = 0;
};
inline void destroy_tensor(ManagedTensor* managed) noexcept {
    if (!managed)
        return;
    auto* owner = static_cast<TensorOwner*>(managed->manager_ctx);
    try {
        DeviceScope scope(owner->device);
        // DLPack aliases may be copied on external CUDA streams. The last
        // alias fences those consumers before unregistering and freeing DMA.
        require_cuda(cudaDeviceSynchronize(), "staging last consumer");
        require_cuda(cudaHostUnregister(owner->buffer->data()), "staging unregister");
    } catch (const std::exception& error) {
        // Completion is unknown: retain the lease and its budget reservation.
        std::fprintf(stderr, "DMA staging retained after cleanup failure: %s\n", error.what());
        return;
    }
    delete owner;
}
inline ManagedTensor* make_tensor(std::shared_ptr<infer::SSDDmaBuffer> buffer, std::size_t bytes,
                                  int device) {
    if (!buffer || !bytes || bytes > buffer->size() || bytes > std::size_t(INT64_MAX))
        throw std::invalid_argument("Invalid native DMA staging tensor");
    auto owner = std::make_unique<TensorOwner>();
    owner->buffer = std::move(buffer);
    owner->shape = bytes;
    owner->device = device;
    DeviceScope scope(device);
    require_cuda(
        cudaHostRegister(owner->buffer->data(), owner->buffer->size(), cudaHostRegisterPortable),
        "staging register");
    owner->managed.dl_tensor = {owner->buffer->data(), {1, 0},  1, {1, 8, 1},
                                &owner->shape,         nullptr, 0};
    owner->managed.manager_ctx = owner.get();
    owner->managed.deleter = destroy_tensor;
    return &owner.release()->managed;
}
} // namespace miniinfer_staging
