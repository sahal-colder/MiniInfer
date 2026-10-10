#include "test_support.h"
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/cache/mini_infer_kv_cache.h"
#include "mini_infer/weights/mini_infer_mmap.h"
#include "kv_test_support.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {
// 只在测试中计数分配/释放和模拟第二次分配失败，不参与框架。
struct AllocationCounts { int allocated = 0, released = 0; };
class CountingAllocator final : public DeviceAllocator {
public:
    explicit CountingAllocator(std::shared_ptr<AllocationCounts> counts, bool fail_second = false)
        : DeviceAllocator(DeviceType::CPU), counts_(std::move(counts)), fail_second_(fail_second) {}
    void* allocate(std::size_t bytes) const override {
        if (fail_second_ && counts_->allocated == 1) throw std::bad_alloc();
        void* ptr = std::malloc(bytes);
        if (!ptr) throw std::bad_alloc();
        ++counts_->allocated;
        return ptr;
    }
    void release(void* ptr) const override { ++counts_->released; std::free(ptr); }
private:
    std::shared_ptr<AllocationCounts> counts_;
    bool fail_second_;
};


// 计数 allocator 验证视图保活、最终释放和第二块分配失败时的清理。
void verify_ownership() {
    std::shared_ptr<AllocationCounts> counts = std::make_shared<AllocationCounts>();
    std::pair<Tensor, Tensor> saved;
    Tensor nested;
    {
        std::shared_ptr<CountingAllocator> allocator = std::make_shared<CountingAllocator>(counts);
        KVCache cache(1, 2, 3, allocator);
        Tensor k = input({3}, 4), v = input({3}, -4);
        cache.append(0, k, v);
        saved = cache.history(0);
        nested = saved.first.view(1, {2});
        cache.reset();
        require(counts->allocated == 2, "views/reset allocated backing storage");
    }
    require(counts->released == 0, "cache destroyed live views' storage");
    equal(saved.first, input({1, 3}, 4));
    saved = {};
    require(counts->released == 1 && nested.index<float>(0) == 5,
            "nested view did not retain root buffer");
    nested = Tensor{};
    require(counts->released == 2, "last view did not release storage");
    std::shared_ptr<AllocationCounts> failure = std::make_shared<AllocationCounts>();
    {
        bool was_rejected = false;
        try {
            KVCache cache(1, 2, 3, std::make_shared<CountingAllocator>(failure, true));
        } catch (const std::bad_alloc&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::bad_alloc");
    }
    require(failure->allocated == 1 && failure->released == 1,
            "failed constructor leaked first buffer");

}

// ModelConfig 只提供尺寸；这里不读取文件，也不创建模型。
void verify_config() {
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    ModelConfig config;
    config.dim = 12; config.head_num = 3; config.kv_head_num = 1;
    config.layer_num = 2; config.seq_len = 4;
    KVCache gqa(config, cpu);
    require(gqa.kv_dim() == 4 && gqa.byte_size() == 2 * 2 * 4 * 4 * sizeof(float),
            "GQA cache must use kv_head_num * head_dim");
    config.kv_head_num = 3;
    KVCache mha(config, cpu);
    require(mha.kv_dim() == 12, "MHA kv_dim must equal dim");
    config.kv_head_num = 2;
    {
        bool was_rejected = false;
        try {
            KVCache invalid(config, cpu);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    config.head_num = 5;
    {
        bool was_rejected = false;
        try {
            KVCache invalid(config, cpu);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
}

// 先同步 GPU，再销毁 cache；剩余历史 view 必须仍可回传读取。
void verify_gpu_view_lifetime(std::shared_ptr<DeviceAllocator> allocator) {
    std::pair<Tensor, Tensor> saved;
    Tensor k = input({3}, 21), v = input({3}, -21);
    Tensor dk = k.to_cuda(allocator), dv = v.to_cuda(allocator);
    {
        KVCache cache(1, 1, 3, allocator);
        cache.append(0, dk, dv);
        wait(DeviceType::CUDA, nullptr);
        saved = cache.history(0);
    }
    equal(saved.first, k.view(0, {1, 3}));
    equal(saved.second, v.view(0, {1, 3}));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        verify_ownership();
        verify_config();
        if (use_cuda) verify_gpu_view_lifetime(std::make_shared<CUDADeviceAllocator>());
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
