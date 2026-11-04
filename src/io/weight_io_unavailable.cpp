#include "infer/io/weight_io.h"
#include <stdexcept>
namespace infer::io {
std::unique_ptr<StorageBackend> make_weight_backend(WeightIOConfig) { throw std::runtime_error("Weight GPU I/O requires CUDA"); }
struct DeviceWeights::Impl {};
DeviceWeights::DeviceWeights(const Safetensors&,StorageBackend&,cudaStream_t,int) { throw std::runtime_error("DeviceWeights requires CUDA"); }
DeviceWeights::~DeviceWeights()=default;
const void* DeviceWeights::data(std::string_view) const { throw std::runtime_error("DeviceWeights requires CUDA"); }
std::size_t DeviceWeights::allocated_bytes() const noexcept { return 0; }
const LoadStats& DeviceWeights::stats() const noexcept { static const LoadStats unavailable{"unavailable",{}};return unavailable; }
}
