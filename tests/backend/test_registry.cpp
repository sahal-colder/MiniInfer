#include "mini_infer/backend/mini_infer_backend.h"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace mini_infer;
int main() {
    try {
        auto& registry = BackendRegistry::instance();
        const BackendKey key{OperatorKind::ADD, DeviceType::CPU, BackendVariant::DEFAULT};
        const auto& saved = registry.find(key);
        std::atomic<bool> failed{false};
        std::vector<std::thread> readers;
        for (int i = 0; i < 4; ++i) readers.emplace_back([&] {
            for (int j = 0; j < 1000; ++j) {
                if (registry.find(key).add != add_kernel_cpu || registry.list().size() < 6) failed = true;
            }
        });
        for (auto variant : {BackendVariant::WARP, BackendVariant::BLOCK, BackendVariant::VECTOR4}) {
            BackendImplementation item;
            item.info = {{OperatorKind::ADD, DeviceType::CPU, variant}, "custom.add"};
            item.add = add_kernel_cpu;
            registry.register_backend(item);
        }
        for (auto& reader : readers) reader.join();
        if (failed || &saved != &registry.find(key))
            throw std::runtime_error("backend registration invalidated an existing lookup");
        if (saved.add != add_kernel_cpu || saved.info.name != "add.cpu")
            throw std::runtime_error("saved backend lookup was corrupted");
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
