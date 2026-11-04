#include "infer/cpu/pinned_memory_pool.h"

namespace infer {

CachingPinnedHostAllocator::CachingPinnedHostAllocator(CachingCPUAllocatorConfig config)
    : DeviceAllocator(DeviceType::CPU), pool_(std::make_shared<PinnedHostAllocator>(), config) {}
CachingPinnedHostAllocator::~CachingPinnedHostAllocator() = default;
void* CachingPinnedHostAllocator::allocate(std::size_t size, cudaStream_t stream) const {
    return pool_.allocate(size, stream);
}
void* CachingPinnedHostAllocator::allocate_io(std::size_t size) const {
    return pool_.allocate_io(size);
}
void CachingPinnedHostAllocator::release(void* ptr, cudaStream_t stream) const {
    pool_.release(ptr, stream);
}
CPUMemoryPoolStats CachingPinnedHostAllocator::stats() const {
    return pool_.stats();
}
void CachingPinnedHostAllocator::trim() const {
    pool_.trim();
}

} // namespace infer
