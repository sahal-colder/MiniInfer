#pragma once

// 第 12 课：算子后端选择。
//
// 前几课直接用 switch(DeviceType) 选择函数。本课把这件事拆成三个
// 可学习的概念：
//   1. BackendKey：一个算子实现的完整身份（算子、设备、变体）；
//   2. BackendRegistry：保存“身份 -> 函数指针”的注册表；
//   3. select_*：从注册表查找实现，并在找不到时给出可读错误。
//
// 这里仍然复用已有 kernel 函数指针类型，所以不会破坏第 1～11 课的 API。

#include "mini_infer/backend/mini_infer_kernels.h"

#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace mini_infer {

enum class OperatorKind {
    ADD,
    RMSNORM
};

// RMSNorm 的变体沿用第 6/8 课的定义；ADD 只使用 DEFAULT。
enum class BackendVariant {
    DEFAULT,
    WARP,
    BLOCK,
    VECTOR4
};

struct BackendKey {
    OperatorKind op = OperatorKind::ADD;
    DeviceType device = DeviceType::UNKNOWN;
    BackendVariant variant = BackendVariant::DEFAULT;

    bool operator==(const BackendKey& other) const;
};

struct BackendInfo {
    BackendKey key;
    std::string name;
};

// 一个注册项同时保存 Add 和 RMSNorm 的函数指针；未使用的指针为空。
struct BackendImplementation {
    BackendInfo info;
    AddKernel add = nullptr;
    RMSNormKernel rmsnorm = nullptr;
};

class BackendRegistry {
public:
    // 返回进程内唯一注册表。默认后端在第一次调用时自动注册。
    static BackendRegistry& instance();

    BackendRegistry(const BackendRegistry&) = delete;
    BackendRegistry& operator=(const BackendRegistry&) = delete;

    // 重复注册同一个 key 会抛异常，避免静默覆盖导致调试困难。
    void register_backend(BackendImplementation implementation);

    const BackendImplementation& find(const BackendKey& key) const;
    std::vector<BackendInfo> list() const;

private:
    BackendRegistry();
    mutable std::mutex mutex_;
    std::deque<BackendImplementation> implementations_;
};

// 显式注册内置 CPU/CUDA 实现；可在示例或程序启动阶段调用。
void register_builtin_backends();

// 便捷选择接口：调用者不需要手动构造 BackendKey。
AddKernel select_add_backend(DeviceType device);
RMSNormKernel select_rmsnorm_backend(
    DeviceType device,
    BackendVariant variant = BackendVariant::BLOCK);

const char* to_string(OperatorKind value);
const char* to_string(DeviceType value);
const char* to_string(BackendVariant value);

}  // namespace mini_infer
