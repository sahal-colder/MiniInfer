#include "test_support.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/cache/mini_infer_kv_cache.h"
#include "kv_test_support.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 从 prefill 到 decode、直接写 slice 和 reset，观察两个层的独立有效长度。
void verify_basic(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    const DeviceType device = allocator->device_type();
    KVCache cache(2, 4, 5, allocator);
    require(cache.byte_size() == 2 * 2 * 4 * 5 * sizeof(float), "KVCache bytes wrong");
    require(cache.history(0).first.empty() && cache.history(0).second.empty(),
            "new cache must have no valid history");
    cache.reset(true, stream);
    wait(device, stream);
    for (int layer = 0; layer < 2; ++layer) {
        all_zero(cache.slice_kv_cache(layer, 0).first);
        all_zero(cache.slice_kv_cache(layer, 3).second);
    }

    Tensor k = input({2, 5}, 1), v = input({2, 5}, -20);
    Tensor dk = on_device(k, allocator), dv = on_device(v, allocator);
    cache.append(0, dk, dv, stream);  // 一次写两个 prompt token。
    wait(device, stream);
    std::pair<Tensor, Tensor> old = cache.history(0);
    equal(old.first, k); equal(old.second, v);
    require(cache.valid_length(0) == 2 && cache.valid_length(1) == 0,
            "per-layer lengths are not independent");
    const float* base = old.first.ptr<float>();
    require(cache.slice_kv_cache(0, 1).first.ptr<float>() == base + 5,
            "token offset should be kv_dim elements");
    require(cache.slice_kv_cache(1, 0).first.ptr<float>() == base + 4 * 5,
            "layer offset should be max_seq_len * kv_dim elements");

    Tensor next_k = input({5}, 101), next_v = input({5}, -101);
    Tensor dnext_k = on_device(next_k, allocator), dnext_v = on_device(next_v, allocator);
    cache.append(0, dnext_k, dnext_v, stream);  // decode 只复制当前 token。
    cache.append(1, dnext_k, dnext_v, stream);
    wait(device, stream);
    equal(cache.slice_kv_cache(0, 2).first, next_k);
    equal(cache.slice_kv_cache(1, 0).second, next_v);
    equal(old.first, k); equal(old.second, v);  // 历史内容保留，旧视图仍为 [2,5]。
    require(cache.history(0).first.ptr<float>() == base, "append reallocated storage");

    // 下一节的 Wk/Wv 算子可以直接把结果写到这里；本例先用 copy_from
    // 展示写 slice 与更新有效长度是两件事。
    std::pair<Tensor, Tensor> last = cache.slice_kv_cache(0, 3);
    last.first.copy_from(dnext_k, stream);
    last.second.copy_from(dnext_v, stream);
    require(cache.valid_length(0) == 3, "slice must not advance valid length");
    cache.commit(0, 3);
    wait(device, stream);
    require(cache.valid_length(0) == 4 && cache.valid_length(1) == 1,
            "commit changed wrong layer");
    equal(last.first, next_k); equal(last.second, next_v);
    {
        bool was_rejected = false;
        try {
            cache.append(0, dnext_k, dnext_v, stream);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    require(cache.valid_length(0) == 4, "failed append changed length");

    cache.reset();
    require(cache.history(0).first.empty() && cache.valid_length(1) == 0,
            "logical reset failed");
    equal(last.first, next_k);  // 逻辑 reset 不清除物理数据。
    cache.append(0, dnext_k, dnext_v, stream);
    wait(device, stream);
    equal(cache.slice_kv_cache(0, 0).first, next_k);
    require(cache.history(0).first.ptr<float>() == base, "reset reallocated storage");
    cache.reset(true, stream);
    wait(device, stream);
    for (int layer = 0; layer < 2; ++layer) {
        for (int pos = 0; pos < 4; ++pos) {
            std::pair<Tensor, Tensor> slot = cache.slice_kv_cache(layer, pos);
            all_zero(slot.first); all_zero(slot.second);
        }
    }
}

// 逐元素核对小宽度、短尾部与触发 grid-stride 的大 prefill。
void verify_sizes(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    for (int32_t dim : {1, 3, 5, 255, 256, 257, 513, 4097}) {
        KVCache cache(2, 3, dim, allocator);
        Tensor k = input({3, dim}, -17), v = input({3, dim}, 29);
        Tensor dk = on_device(k, allocator), dv = on_device(v, allocator);
        cache.append(0, dk, dv, stream);
        cache.append(1, dv, dk, stream);
        wait(allocator->device_type(), stream);
        equal(cache.history(0).first, k); equal(cache.history(0).second, v);
        equal(cache.history(1).first, v); equal(cache.history(1).second, k);
    }
    // 4097*257 > 4096*256，触发 CUDA grid-stride 的第二轮及短尾部。
    KVCache large(1, 4097, 257, allocator);
    Tensor k = input({4097, 257}, -300), v = input({4097, 257}, 800);
    Tensor dk = on_device(k, allocator), dv = on_device(v, allocator);
    large.append(0, dk, dv, stream);
    wait(allocator->device_type(), stream);
    equal(large.history(0).first, k); equal(large.history(0).second, v);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
        verify_basic(cpu, nullptr);
        verify_sizes(cpu, nullptr);
        if (use_cuda) {
            std::shared_ptr<CUDADeviceAllocator> gpu = std::make_shared<CUDADeviceAllocator>();
            TestStream stream;
            verify_basic(gpu, nullptr);
            verify_basic(gpu, stream.get());
            verify_sizes(gpu, nullptr);
            verify_sizes(gpu, stream.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
