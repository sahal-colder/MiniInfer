#include "test_support.h"
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "mini_infer/cache/mini_infer_kv_cache.h"
#include "kv_test_support.h"

using namespace mini_infer;
using namespace mini_infer_test;
namespace {

// 输入非法时，先拒绝再写入；每一组使用独立缓存。
void verify_index_rejections(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    KVCache cache(2, 2, 3, allocator);
    Tensor k = on_device(input({3}, 1), allocator);
    Tensor v = on_device(input({3}, 9), allocator);
    {
        bool was_rejected = false;
        try {
            cache.slice_kv_cache(-1, 0);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.slice_kv_cache(2, 0);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.slice_kv_cache(0, -1);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.slice_kv_cache(0, 2);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.history(2);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.append(-1, k, v, stream);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            cache.commit(0, 1);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            cache.commit(0, 0, 0);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            cache.commit(0, 0, 3);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    require(cache.valid_length(0) == 0, "rejected call changed the valid prefix");
}

// 输入非法时，先拒绝再写入；每一组使用独立缓存。
void verify_input_rejections(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    KVCache cache(2, 2, 3, allocator);
    Tensor k = on_device(input({3}, 1), allocator);
    Tensor v = on_device(input({3}, 9), allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, Tensor{}, v, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor no_storage(DataType::FLOAT32, {3});
    {
        bool was_rejected = false;
        try {
            cache.append(0, no_storage, v, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor bad_shape = on_device(input({1, 3}, 1), allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, bad_shape, v, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor bad_dim = on_device(input({2}, 1), allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, bad_dim, bad_dim, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor bad_rank = on_device(input({1, 1, 3}, 1), allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, bad_rank, bad_rank, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor integer(DataType::INT32, {3}, allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, integer, integer, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    Tensor batch = on_device(input({3, 3}, 1), allocator);
    {
        bool was_rejected = false;
        try {
            cache.append(0, batch, batch, stream);
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    require(cache.valid_length(0) == 0, "rejected call changed the valid prefix");
}

// 输入非法时，先拒绝再写入；每一组使用独立缓存。
void verify_alias_rejections(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    KVCache cache(2, 2, 3, allocator);
    Tensor k = on_device(input({3}, 1), allocator);
    Tensor v = on_device(input({3}, 9), allocator);
    std::pair<Tensor, Tensor> slot = cache.slice_kv_cache(0, 0);
    {
        bool was_rejected = false;
        try {
            cache.append(0, slot.first, v, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    KVCacheWriteKernel kernel = get_kv_cache_write_kernel(allocator->device_type());
    {
        bool was_rejected = false;
        try {
            kernel(k, v, slot.first, slot.first, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            kernel(slot.first, v, slot.first, slot.second, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            kernel(k, v, slot.first, v, stream);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    if (allocator->device_type() == DeviceType::CUDA) {
        Tensor host = input({3}, 1);
        {
            bool was_rejected = false;
            try {
                cache.append(0, host, host, stream);
            } catch (const std::invalid_argument&) {
                was_rejected = true;
            }
            require(was_rejected, "expected std::invalid_argument");
        }
        {
            bool was_rejected = false;
            try {
                kv_cache_write_cpu(k, v, slot.first, slot.second, stream);
            } catch (const std::invalid_argument&) {
                was_rejected = true;
            }
            require(was_rejected, "expected std::invalid_argument");
        }
    } else {
        // 用主机地址标注 CUDA，仅检查元数据，绝不解引用或提交 kernel。
        Tensor wrong_device = Tensor::borrow(DataType::FLOAT32, {3}, k.ptr<float>(), DeviceType::CUDA);
        {
            bool was_rejected = false;
            try {
                cache.append(0, wrong_device, wrong_device, stream);
            } catch (const std::invalid_argument&) {
                was_rejected = true;
            }
            require(was_rejected, "expected std::invalid_argument");
        }
        {
            bool was_rejected = false;
            try {
                kv_cache_write_cuda(k, v, slot.first, slot.second, stream);
            } catch (const std::invalid_argument&) {
                was_rejected = true;
            }
            require(was_rejected, "expected std::invalid_argument");
        }
    }
    require(cache.valid_length(0) == 0 && cache.valid_length(1) == 0,
            "rejected write modified valid lengths");
    cache.append(0, k, v, stream);
    wait(allocator->device_type(), stream);
    equal(cache.history(0).first, k.view(0, {1, 3}).to_cpu());
    {
        bool was_rejected = false;
        try {
            cache.commit(0, 0);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            cache.commit(0, 1, std::numeric_limits<int32_t>::max());
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }

}

// 输入非法时，先拒绝再写入；每一组使用独立缓存。
void verify_constructor_rejections(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    Tensor k = on_device(input({3}, 1), allocator);
    (void)stream;
    std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
    {
        bool was_rejected = false;
        try {
            KVCache invalid(0, 1, 1, cpu);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            KVCache invalid(1, -1, 1, cpu);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            KVCache invalid(1, 1, 0, cpu);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            KVCache invalid(1, 1, 1, nullptr);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            KVCache invalid(std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(), 3, cpu);
        } catch (const std::overflow_error&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::overflow_error");
    }
    // 单块字节数可表达但两块合计溢出，同样在真实分配前拒绝。
    {
        bool was_rejected = false;
        try {
            KVCache invalid(std::numeric_limits<int32_t>::max(),
            std::numeric_limits<int32_t>::max(), 1, cpu);
        } catch (const std::overflow_error&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::overflow_error");
    }
    {
        bool was_rejected = false;
        try {
            get_kv_cache_write_kernel(DeviceType::UNKNOWN);
        } catch (const std::invalid_argument&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::invalid_argument");
    }
    {
        bool was_rejected = false;
        try {
            k.view(2, {2});
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            k.view(std::numeric_limits<std::size_t>::max(), {1});
        } catch (const std::out_of_range&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::out_of_range");
    }
    {
        bool was_rejected = false;
        try {
            Tensor{}.view(0, {1});
        } catch (const std::logic_error&) {
            was_rejected = true;
        }
        require(was_rejected, "expected std::logic_error");
    }
}

void verify_rejections(std::shared_ptr<DeviceAllocator> allocator, cudaStream_t stream) {
    verify_index_rejections(allocator, stream);
    verify_input_rejections(allocator, stream);
    verify_alias_rejections(allocator, stream);
    verify_constructor_rejections(allocator, stream);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const bool use_cuda = test_uses_cuda(argc, argv);
        std::shared_ptr<CPUDeviceAllocator> cpu = std::make_shared<CPUDeviceAllocator>();
        verify_rejections(cpu, nullptr);
        if (use_cuda) {
            std::shared_ptr<CUDADeviceAllocator> gpu = std::make_shared<CUDADeviceAllocator>();
            TestStream stream;
            verify_rejections(gpu, nullptr);
            verify_rejections(gpu, stream.get());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
