#include "mini_infer/backend/mini_infer_backend.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mini_infer {
namespace {

BackendImplementation add_cpu() {
    BackendImplementation result;
    result.info = {{OperatorKind::ADD, DeviceType::CPU, BackendVariant::DEFAULT}, "add.cpu"};
    result.add = add_kernel_cpu;
    return result;
}

BackendImplementation add_cuda() {
    BackendImplementation result;
    result.info = {{OperatorKind::ADD, DeviceType::CUDA, BackendVariant::DEFAULT}, "add.cuda"};
    result.add = add_kernel_cuda;
    return result;
}

BackendImplementation rmsnorm_cpu() {
    BackendImplementation result;
    result.info = {{OperatorKind::RMSNORM, DeviceType::CPU, BackendVariant::DEFAULT}, "rmsnorm.cpu"};
    result.rmsnorm = rmsnorm_kernel_cpu;
    return result;
}

BackendImplementation rmsnorm_cuda(BackendVariant variant, RMSNormKernel kernel,
                                   const char* name) {
    BackendImplementation result;
    result.info = {{OperatorKind::RMSNORM, DeviceType::CUDA, variant}, name};
    result.rmsnorm = kernel;
    return result;
}

std::string key_string(const BackendKey& key) {
    std::ostringstream out;
    out << to_string(key.op) << "." << to_string(key.device) << "."
        << to_string(key.variant);
    return out.str();
}

}  // namespace

bool BackendKey::operator==(const BackendKey& other) const {
    return op == other.op && device == other.device && variant == other.variant;
}

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

BackendRegistry::BackendRegistry() {
    // 构造函数中直接填充，避免 instance() -> constructor -> instance()
    // 的静态初始化递归。显式 register_builtin_backends() 会保持幂等。
    register_backend(add_cpu());
    register_backend(add_cuda());
    register_backend(rmsnorm_cpu());
    register_backend(rmsnorm_cuda(BackendVariant::WARP,
                                  rmsnorm_kernel_cuda_warp,
                                  "rmsnorm.cuda.warp"));
    register_backend(rmsnorm_cuda(BackendVariant::BLOCK,
                                  rmsnorm_kernel_cuda,
                                  "rmsnorm.cuda.block"));
    register_backend(rmsnorm_cuda(BackendVariant::VECTOR4,
                                  rmsnorm_kernel_cuda_vectorized,
                                  "rmsnorm.cuda.vector4"));
}

void BackendRegistry::register_backend(BackendImplementation implementation) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (implementation.info.name.empty()) {
        throw std::invalid_argument("backend name must not be empty");
    }
    if ((implementation.info.key.op == OperatorKind::ADD) !=
        (implementation.add != nullptr)) {
        throw std::invalid_argument("ADD backend must provide exactly an Add kernel");
    }
    if ((implementation.info.key.op == OperatorKind::RMSNORM) !=
        (implementation.rmsnorm != nullptr)) {
        throw std::invalid_argument("RMSNorm backend must provide exactly an RMSNorm kernel");
    }
    const auto duplicate = std::find_if(
        implementations_.begin(), implementations_.end(),
        [&implementation](const BackendImplementation& current) {
            return current.info.key == implementation.info.key;
        });
    if (duplicate != implementations_.end()) {
        throw std::invalid_argument("backend already registered: " +
                                    key_string(implementation.info.key));
    }
    implementations_.push_back(std::move(implementation));
}

const BackendImplementation& BackendRegistry::find(const BackendKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = std::find_if(
        implementations_.begin(), implementations_.end(),
        [&key](const BackendImplementation& current) {
            return current.info.key == key;
        });
    if (it == implementations_.end()) {
        throw std::invalid_argument("backend not found: " + key_string(key));
    }
    return *it;
}

std::vector<BackendInfo> BackendRegistry::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<BackendInfo> result;
    result.reserve(implementations_.size());
    for (const auto& implementation : implementations_) result.push_back(implementation.info);
    return result;
}

void register_builtin_backends() {
    auto& registry = BackendRegistry::instance();
    // 内置后端在构造函数中已注册；该函数是给课程示例的显式初始化入口。
    // 保留它可以让未来的动态插件注册拥有稳定的启动位置。
    (void)registry;
}

AddKernel select_add_backend(DeviceType device) {
    return BackendRegistry::instance()
        .find({OperatorKind::ADD, device, BackendVariant::DEFAULT})
        .add;
}

RMSNormKernel select_rmsnorm_backend(DeviceType device, BackendVariant variant) {
    const auto selected_variant = device == DeviceType::CPU
                                      ? BackendVariant::DEFAULT
                                      : variant;
    return BackendRegistry::instance()
        .find({OperatorKind::RMSNORM, device, selected_variant})
        .rmsnorm;
}

const char* to_string(OperatorKind value) {
    switch (value) {
        case OperatorKind::ADD: return "add";
        case OperatorKind::RMSNORM: return "rmsnorm";
    }
    return "unknown-op";
}

const char* to_string(DeviceType value) {
    switch (value) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::CUDA: return "cuda";
        case DeviceType::UNKNOWN: return "unknown-device";
    }
    return "unknown-device";
}

const char* to_string(BackendVariant value) {
    switch (value) {
        case BackendVariant::DEFAULT: return "default";
        case BackendVariant::WARP: return "warp";
        case BackendVariant::BLOCK: return "block";
        case BackendVariant::VECTOR4: return "vector4";
    }
    return "unknown-variant";
}

}  // namespace mini_infer
