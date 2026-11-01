#include "infer/storage/storage_backend.h"
#include <system_error>

namespace infer {
namespace {
[[noreturn]] void unavailable() {
    throw std::system_error(std::make_error_code(std::errc::operation_not_supported),
                            "GPU storage requires CUDA support");
}
} // namespace
struct GPUStorageBackend::Impl {};
GPUStorageBackend::GPUStorageBackend(std::size_t, std::size_t, int) { unavailable(); }
GPUStorageBackend::~GPUStorageBackend() = default;
std::size_t GPUStorageBackend::slot_bytes() const noexcept { return 0; }
StorageHandle GPUStorageBackend::reserve() { unavailable(); }
void GPUStorageBackend::release(StorageHandle) { unavailable(); }
StorageMemoryView GPUStorageBackend::memory_view(StorageHandle) const { unavailable(); }
void GPUStorageBackend::read(StorageHandle, std::size_t, void*, std::size_t) { unavailable(); }
void GPUStorageBackend::write(StorageHandle, std::size_t, const void*, std::size_t) { unavailable(); }
void GPUStorageBackend::validate_stream(cudaStream_t) const { unavailable(); }
void GPUStorageBackend::wait(cudaStream_t) const { unavailable(); }
void GPUStorageBackend::wait_transfer() const { unavailable(); }
BackendPoolStats GPUStorageBackend::pool_stats() const { unavailable(); }
} // namespace infer
