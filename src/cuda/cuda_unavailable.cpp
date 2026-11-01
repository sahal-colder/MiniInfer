#include "infer/cuda/cuda_memory_pool.h"
#include <system_error>

namespace infer {
namespace {
[[noreturn]] void unavailable() {
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "CUDA support is disabled");
}
} // namespace
class CachingCUDADeviceAllocator::Impl {};
CachingCUDADeviceAllocator::CachingCUDADeviceAllocator(CachingCUDAAllocatorConfig)
    : DeviceAllocator(DeviceType::CUDA) { unavailable(); }
CachingCUDADeviceAllocator::~CachingCUDADeviceAllocator() = default;
void* CachingCUDADeviceAllocator::allocate(std::size_t, cudaStream_t) const { unavailable(); }
void CachingCUDADeviceAllocator::release(void*, cudaStream_t) const { unavailable(); }
CachingCUDADeviceAllocator::StreamId
CachingCUDADeviceAllocator::register_stream(cudaStream_t, StreamId) const { unavailable(); }
void CachingCUDADeviceAllocator::unregister_stream(cudaStream_t) const { unavailable(); }
void CachingCUDADeviceAllocator::reserve_graph_workspace(std::size_t) const { unavailable(); }
void CachingCUDADeviceAllocator::begin_graph_capture(GraphId, cudaStream_t) const { unavailable(); }
void CachingCUDADeviceAllocator::end_graph_capture(GraphId) const { unavailable(); }
void CachingCUDADeviceAllocator::abort_graph_capture(GraphId) const { unavailable(); }
std::size_t CachingCUDADeviceAllocator::graph_workspace_used(GraphId) const { unavailable(); }
void CachingCUDADeviceAllocator::forget_graph(GraphId) const { unavailable(); }
void CachingCUDADeviceAllocator::release_graph_workspace() const { unavailable(); }
void CachingCUDADeviceAllocator::flush() const { unavailable(); }
void CachingCUDADeviceAllocator::collect() const { unavailable(); }
void CachingCUDADeviceAllocator::trim() const { unavailable(); }
CUDAMemoryPoolStats CachingCUDADeviceAllocator::stats() const { unavailable(); }
} // namespace infer
