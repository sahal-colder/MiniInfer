#pragma once
// #pragma once：
// 防止这个头文件被重复包含。
//
// 比如：
// a.cpp -> include tensor.h
//       -> include xxx.h -> 又 include tensor.h
//
// 如果没有头文件保护，就可能导致 Tensor 类被重复定义。
// #pragma once 可以保证当前头文件只被编译一次。


#include "mini_infer/core/mini_infer_memory.h"
// 引入我们前面写的内存管理模块。
//
// 这个文件里面应该定义了：
//
// DeviceType
// DeviceAllocator
// Buffer
//
// Tensor 本身并不直接负责 malloc/cudaMalloc，
// 而是通过 DeviceAllocator + Buffer 管理内存。


#include <cstdint>
// 提供固定长度整数类型：
// int32_t
// uint32_t
// int64_t
// 等。


#include <stdexcept>
// 提供 C++ 标准异常，例如：
//
// std::invalid_argument
// std::runtime_error
//
// 后面 Tensor 会通过 throw 抛出错误。


#include <type_traits>
// 用于编译期类型判断。
//
// 后面会使用：
//
// std::is_same_v<T, float>
//
// 判断模板参数 T 是不是 float。


#include <vector>
// Tensor 的 shape 使用 vector 保存。
//
// 例如：
// Tensor shape = [2, 3, 4]
//
// 在代码里面就是：
//
// std::vector<int32_t>{2, 3, 4}


namespace mini_infer {

// ============================================================
// 1. Tensor 支持的数据类型
// ============================================================

enum class DataType {
    UNKNOWN = 0,
    // UNKNOWN 表示数据类型还没有设置。
    //
    // 默认构造：
    //
    // Tensor tensor;
    //
    // 这时候 Tensor 还没有真正绑定数据，
    // 所以 dtype 就是 UNKNOWN。


    FLOAT32,
    // 32 位浮点数。
    //
    // 对应 C++：
    //
    // float
    //
    // 一个 float 通常占 4 字节。


    INT32,
    // 32 位整数。
    //
    // 对应：
    //
    // int32_t
    //
    // 一个 int32_t 占 4 字节。
    // 8 位有符号整数。第 7 课的对称量化权重使用这个类型。
    // 量化权重不是“可以直接交给现有 FP32 Layer 的输入”，
    // 而是由量化算子按 group scale 恢复后参与计算。
    INT8
};


// ============================================================
// 根据 DataType 返回一个元素占多少字节
// ============================================================

std::size_t data_type_size(DataType dtype);

// 这里只是函数声明，实现应该写在 .cu / .cpp 文件中。
//
// 例如可能是：
//
// std::size_t data_type_size(DataType dtype) {
//     switch (dtype) {
//         case DataType::FLOAT32:
//             return sizeof(float);
//
//         case DataType::INT32:
//             return sizeof(int32_t);
//
//         default:
//             throw std::invalid_argument("unknown dtype");
//     }
// }
//
// 例如：
//
// FLOAT32 -> 4 字节
// INT32   -> 4 字节



// ============================================================
// 2. Tensor 类
// ============================================================
//
// Tensor 可以理解成：
//
//             Tensor
//               │
//        ┌──────┼──────────┐
//        │      │          │
//      shape   dtype     Buffer
//        │      │          │
//     [2,3]   float     真正的数据
//
//
//
// Tensor 本身主要保存的是“描述信息”。
// 真正的大块内存由 Buffer 管理。
//
// 例如：
//
// Tensor<float>：
//
// dims_   = {2, 3}
// size_   = 6
// dtype_  = FLOAT32
// buffer_ -----> Buffer -----> CPU/CUDA 内存
//
//
// ============================================================
//
// 这里有一个非常重要的设计：
//
// std::shared_ptr<Buffer> buffer_;
//
// 所以：
//
// Tensor a = b;
//
// 默认不会复制真实数据。
//
// 而是：
//
// a.buffer_ ─────┐
//                ├──> 同一个 Buffer
// b.buffer_ ─────┘
//
// 也就是共享数据。
//
// 真正想复制数据时：
//
// Tensor c = b.clone(...);
//
// 才会重新申请一块内存并拷贝数据。
//
// 这和 PyTorch Tensor 的一些内部设计思想非常像。
// ============================================================


class Tensor {

public:

    // ========================================================
    // 3. 默认构造函数
    // ========================================================

    Tensor() = default;
    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // 什么都不做。
    //
    // 例如：
    //
    // Tensor x;
    //
    // 此时：
    //
    // dtype_  = UNKNOWN
    // dims_   = {}
    // size_   = 0
    // buffer_ = nullptr
    //
    //
    // 这种 Tensor 可以理解为：
    //
    // “空 Tensor”
    // 或
    // “还没有绑定数据的 Tensor”
    //
    //
    // 为什么需要空 Tensor？
    //
    // 比如神经网络 Operator：
    //
    // class Linear {
    //     Tensor input_;
    //     Tensor output_;
    // };
    //
    // 一开始创建 Linear 时，
    // input/output 可能还不知道是什么，
    // 所以先创建空 Tensor。


    // ========================================================
    // 4. Tensor 构造函数
    // ========================================================

    Tensor(
        DataType dtype,
        std::vector<int32_t> dims,
        std::shared_ptr<DeviceAllocator> allocator = nullptr
    );

    /*
     * 参数 1：
     *
     * DataType dtype
     *
     * Tensor 的元素类型。
     *
     * 例如：
     *
     * DataType::FLOAT32
     *
     *
     * 参数 2：
     *
     * std::vector<int32_t> dims
     *
     * Tensor 的形状。
     *
     * 例如：
     *
     * {2, 3}
     *
     * 表示：
     *
     * 2 行 3 列
     *
     *
     * 参数 3：
     *
     * std::shared_ptr<DeviceAllocator> allocator
     *
     * 指定使用哪个内存分配器。
     *
     * 例如：
     *
     * CPUAllocator
     * CUDAAllocator
     *
     *
     * 如果 allocator != nullptr：
     *
     * 通常构造的时候就可以直接申请内存。
     *
     *
     * 举例：
     *
     * Tensor x(
     *     DataType::FLOAT32,
     *     {2, 3},
     *     cpu_allocator
     * );
     *
     *
     * 那么：
     *
     * shape = 2 × 3
     *
     * 元素数量：
     *
     * 2 × 3 = 6
     *
     * 每个 float：
     *
     * 4 字节
     *
     * 总内存：
     *
     * 6 × 4 = 24 字节
     */


    // ========================================================
    // 5. borrow：借用别人已经申请好的内存
    // ========================================================

    static Tensor borrow(
        DataType dtype,
        std::vector<int32_t> dims,
        void* ptr,
        DeviceType device
    );

    // 第 15 课需要把大 KV Cache 中的一段封装为连续 Tensor。
    // offset 按元素计数，dims 指定新形状；不分配/复制实际数据。
    // 与 borrow 不同，view 会保活当前 Buffer：原 Tensor 析构后，
    // 自有存储仍然有效。若原 Tensor 借用外部内存，外部 owner 的
    // 生命周期依然由调用者保证。view 共享数据，写入会修改原存储。
    Tensor view(std::size_t offset, std::vector<int32_t> dims) const;

    /*
     * borrow 的意思是：
     *
     * Tensor 不自己申请内存，
     * 而是直接使用别人已经存在的内存。
     *
     *
     * 举例：
     *
     * float data[6] = {
     *     1, 2, 3,
     *     4, 5, 6
     * };
     *
     * Tensor x = Tensor::borrow(
     *     DataType::FLOAT32,
     *     {2, 3},
     *     data,
     *     DeviceType::CPU
     * );
     *
     *
     * 此时：
     *
     * x
     * │
     * buffer_
     * │
     * └──────────────> data
     *
     *
     * Tensor 没有重新 malloc。
     *
     *
     * ------------------------------
     * 为什么叫“借用”？
     * ------------------------------
     *
     * 因为这块内存并不一定属于 Tensor。
     *
     * 所以调用方必须保证：
     *
     * 1. 内存足够大
     * 2. 内存地址合法
     * 3. CPU/CUDA device 参数正确
     * 4. Tensor 使用期间，这块内存不能提前释放
     *
     *
     * 比如下面就是危险的：
     *
     * Tensor create_tensor() {
     *
     *     float data[10];
     *
     *     return Tensor::borrow(..., data, ...);
     * }
     *
     *
     * 函数退出以后：
     *
     * data 已经销毁了。
     *
     * Tensor 里面的 ptr 就变成悬空指针。
     */


    // ========================================================
    // 6. 给 Tensor 分配真正的存储空间
    // ========================================================

    void allocate(std::shared_ptr<DeviceAllocator> allocator);

    /*
     * 如果 Tensor 只有：
     *
     * dtype
     * shape
     *
     * 但是还没有 Buffer，
     *
     * 就可以通过 allocate() 分配内存。
     *
     *
     * 例如：
     *
     * Tensor x(
     *     DataType::FLOAT32,
     *     {2, 3}
     * );
     *
     * 此时可能：
     *
     * buffer_ == nullptr
     *
     *
     * 再执行：
     *
     * x.allocate(cpu_allocator);
     *
     *
     * 内部可能最终执行：
     *
     * allocator->allocate(24);
     *
     * 获得 24 字节内存。
     */


    // ========================================================
    // 7. size()：Tensor 有多少个元素
    // ========================================================

    std::size_t size() const {
        return size_;
    }

    /*
     * 注意：
     *
     * size() 返回的是：
     *
     * 元素数量
     *
     * 不是字节数。
     *
     *
     * 例如 shape：
     *
     * [2, 3, 4]
     *
     * 那么：
     *
     * size_ = 2 × 3 × 4
     *       = 24
     */


    // ========================================================
    // 8. byte_size()：Tensor 一共占多少字节
    // ========================================================

    std::size_t byte_size() const;

    /*
     * 一般：
     *
     * byte_size
     *
     * =
     *
     * size()
     *
     * ×
     *
     * data_type_size(dtype)
     *
     *
     * 例如：
     *
     * shape = [2, 3]
     * dtype = FLOAT32
     *
     * size = 6
     *
     * float = 4 字节
     *
     * byte_size = 6 × 4
     *           = 24 字节
     */


    // ========================================================
    // 9. 获取数据类型
    // ========================================================

    DataType data_type() const {
        return dtype_;
    }


    // ========================================================
    // 10. 获取 Tensor 在哪个设备
    // ========================================================

    DeviceType device_type() const;

    /*
     * 返回例如：
     *
     * DeviceType::CPU
     *
     * 或：
     *
     * DeviceType::CUDA
     *
     *
     * 一般会通过 buffer_ 查询。
     *
     * 类似：
     *
     * return buffer_->device_type();
     */


    // ========================================================
    // 11. Tensor 是否没有真正的存储空间
    // ========================================================

    bool empty() const {
        return !buffer_;
    }

    /*
     * std::shared_ptr 可以直接转 bool。
     *
     *
     * buffer_ == nullptr
     *
     * 那么：
     *
     * !buffer_ == true
     *
     * 所以 empty() == true。
     *
     *
     * 注意：
     *
     * 这里的 empty 并不是：
     *
     * size == 0
     *
     * 而是：
     *
     * 没有 Buffer。
     */


    // ========================================================
    // 12. 获取 Tensor 的 shape
    // ========================================================

    const std::vector<int32_t>& dims() const {
        return dims_;
    }

    /*
     * 例如：
     *
     * Tensor shape = [2,3,4]
     *
     * 那么：
     *
     * tensor.dims()
     *
     * 返回：
     *
     * {2, 3, 4}
     *
     *
     * 为什么返回：
     *
     * const std::vector<int32_t>&
     *
     * 而不是：
     *
     * std::vector<int32_t>
     *
     *
     * 因为返回引用不会复制整个 vector。
     *
     * const 又保证外部不能随便修改：
     *
     * tensor.dims()[0] = 100;
     *
     * 这种操作会被禁止。
     */


    // ========================================================
    // 13. 获取第 i 个维度
    // ========================================================

    int32_t dim(std::size_t i) const {
        return dims_.at(i);
    }

    /*
     * 例如：
     *
     * shape = [2, 3, 4]
     *
     * dim(0) = 2
     * dim(1) = 3
     * dim(2) = 4
     *
     *
     * 为什么使用 at() 而不是 []？
     *
     * 因为：
     *
     * dims_.at(i)
     *
     * 会检查数组越界。
     *
     *
     * 如果：
     *
     * dim(100)
     *
     * 会抛异常。
     */


    // ========================================================
    // 14. 计算 strides
    // ========================================================

    std::vector<std::size_t> strides() const;

    /*
     * stride 中文一般叫：
     *
     * 步长
     *
     *
     * 举例：
     *
     * shape = [2, 3, 4]
     *
     * 连续、行优先存储。
     *
     *
     * 数据布局：
     *
     * 第 2 维变化一次：
     *
     * offset + 1
     *
     *
     * 第 1 维变化一次：
     *
     * 需要跨过 4 个元素
     *
     *
     * 第 0 维变化一次：
     *
     * 需要跨过 3 × 4 = 12 个元素
     *
     *
     * 所以：
     *
     * stride = [12, 4, 1]
     *
     *
     * 如果访问：
     *
     * tensor[i][j][k]
     *
     * 内存 offset：
     *
     * i * 12
     * +
     * j * 4
     * +
     * k * 1
     */


    // ========================================================
    // 15. reshape
    // ========================================================

    void reshape(std::vector<int32_t> dims);

    /*
     * reshape：
     *
     * 改变 Tensor 的形状，
     * 但是不改变真实数据。
     *
     *
     * 例如：
     *
     * 原来：
     *
     * [2, 3]
     *
     * 一共：
     *
     * 6 个元素
     *
     *
     * 可以变成：
     *
     * [3, 2]
     *
     * 因为：
     *
     * 3 × 2 = 6
     *
     *
     * 也可以：
     *
     * [6]
     *
     *
     * 但是不能：
     *
     * [4, 2]
     *
     * 因为：
     *
     * 4 × 2 = 8
     *
     * 元素数量变了。
     */


    // ========================================================
    // 16. ptr<T>()：取得底层数据指针
    // ========================================================

    template <typename T>
    T* ptr() {

        // 第一步：
        // 检查用户要求的 T 和 Tensor 的 dtype 是否一致。
        //
        // 例如：
        //
        // Tensor 实际是 FLOAT32
        //
        // tensor.ptr<float>();
        //
        // 正确。
        //
        //
        // tensor.ptr<int32_t>();
        //
        // 错误。
        check_type<T>();


        // 第二步：
        // 检查 Tensor 是否真的有 Buffer。
        require_storage();


        // 第三步：
        // buffer_->ptr() 通常返回 void*
        //
        // static_cast<T*> 转换成真正的类型。
        return static_cast<T*>(buffer_->ptr());
    }


    // const 版本
    template <typename T>
    const T* ptr() const {

        check_type<T>();

        require_storage();

        return static_cast<const T*>(buffer_->ptr());
    }

    /*
     * 为什么需要两个 ptr？
     *
     *
     * 普通 Tensor：
     *
     * Tensor x;
     *
     * float* p = x.ptr<float>();
     *
     * 可以修改：
     *
     * p[0] = 100;
     *
     *
     * ------------------------------
     *
     * const Tensor：
     *
     * const Tensor x;
     *
     * const float* p = x.ptr<float>();
     *
     * 只能读。
     */


    // ========================================================
    // 17. index()：通过 offset 访问某一个元素
    // ========================================================

    template <typename T>
    T& index(std::size_t offset) {

        // 检查 offset 是否越界。
        //
        // 比如：
        //
        // Tensor 一共 6 个元素
        //
        // 合法：
        // 0 ~ 5
        //
        // index(6)
        // 就越界。
        check_index(offset);


        // ptr<T>() 得到首地址。
        //
        // [offset] 访问指定元素。
        return ptr<T>()[offset];
    }


    template <typename T>
    const T& index(std::size_t offset) const {

        check_index(offset);

        return ptr<T>()[offset];
    }


    /*
     * 举例：
     *
     * Tensor x FLOAT32
     *
     * 数据：
     *
     * [10, 20, 30, 40]
     *
     *
     * x.index<float>(0)
     *
     * 得到：
     *
     * 10
     *
     *
     * x.index<float>(2)
     *
     * 得到：
     *
     * 30
     *
     *
     * ------------------------------
     *
     * 特别重要：
     *
     * index() 只适合 CPU Tensor。
     *
     *
     * 因为如果 ptr() 返回的是：
     *
     * CUDA device pointer
     *
     * CPU 不能直接：
     *
     * ptr[0]
     *
     * 去解引用 CUDA 显存。
     *
     *
     * CUDA Tensor 一般需要：
     *
     * cudaMemcpy
     *
     * 或 CUDA kernel
     *
     * 才能访问。
     */


    // ========================================================
    // 18. copy_from：从另一个 Tensor 拷贝数据
    // ========================================================

    void copy_from(
        const Tensor& source,
        cudaStream_t stream = nullptr
    );

    /*
     * 假设：
     *
     * Tensor A
     *
     * Tensor B
     *
     * B.copy_from(A);
     *
     *
     * 意思：
     *
     * 把 A 的数据复制到 B。
     *
     *
     * 根据设备不同，底层可能调用：
     *
     * CPU -> CPU
     * memcpy
     *
     * CPU -> CUDA
     * cudaMemcpy
     *
     * CUDA -> CPU
     * cudaMemcpy
     *
     * CUDA -> CUDA
     * cudaMemcpy
     *
     *
     * ---------------------------------
     * stream 参数
     * ---------------------------------
     *
     * stream == nullptr
     *
     * 表示同步复制。
     *
     * 函数返回时数据通常已经准备好了。
     *
     *
     * 指定 CUDA stream：
     *
     * copy_from(source, stream);
     *
     * 可以使用异步 CUDA copy。
     *
     *
     * 例如：
     *
     * H2D copy
     * ↓
     * forward
     * ↓
     * D2H copy
     *
     * 都提交到同一个 stream。
     *
     * CUDA 会按照顺序执行。
     */


    // ========================================================
    // 19. clone：深拷贝
    // ========================================================

    Tensor clone(
        std::shared_ptr<DeviceAllocator> allocator
    ) const;

    // ========================================================
    // Lesson 9: Tensor 的 CPU / CUDA 迁移
    // ========================================================
    //
    // 课程原代码直接替换当前 Tensor 的 buffer_。这里改成返回新 Tensor：
    //
    //   Tensor device = host.to_cuda(cuda_pool);
    //
    // 这样 host 仍然有效，也不会因为异步 stream 尚未执行完就销毁旧
    // Buffer。若已经在目标设备上，则返回共享同一 Buffer 的浅拷贝。
    //
    // 使用非空 stream 时拷贝是异步的；调用者必须保证源和目标 Tensor
    // 一直存活，并在 CPU 读取结果或释放内存前同步该 stream。
    Tensor to_cuda(
        std::shared_ptr<DeviceAllocator> cuda_allocator,
        cudaStream_t stream = nullptr
    ) const;

    Tensor to_cpu(cudaStream_t stream = nullptr) const;

    /*
     * clone 和普通赋值完全不同。
     *
     *
     * ------------------------------
     * 普通复制
     * ------------------------------
     *
     * Tensor b = a;
     *
     *
     * a.buffer_
     *      │
     *      └──────┐
     *             ↓
     *           Buffer
     *             ↑
     *      ┌──────┘
     * b.buffer_
     *
     *
     * a 和 b 共用同一块数据。
     *
     *
     * ------------------------------
     * clone
     * ------------------------------
     *
     * Tensor b = a.clone(allocator);
     *
     *
     * a.buffer_
     *    │
     *    ↓
     * Buffer A
     *
     *
     * b.buffer_
     *    │
     *    ↓
     * Buffer B
     *
     *
     * 数据内容一样，
     * 但是内存已经完全独立。
     */


    // ========================================================
    // 20. 检查两个 Tensor 的内存是否重叠
    // ========================================================

    bool overlaps(const Tensor& other) const;

    /*
     * 判断：
     *
     * 当前 Tensor 使用的内存区域
     *
     * 和
     *
     * other 使用的内存区域
     *
     * 有没有重叠。
     *
     *
     * 为什么需要这个函数？
     *
     * 比如：
     *
     * Tensor A、B 实际指向同一块内存。
     *
     * 然后：
     *
     * A.copy_from(B);
     *
     * 某些复制操作就需要特别处理。
     *
     *
     * 这个功能以后处理：
     *
     * inplace
     * alias
     * memory reuse
     *
     * 都很有用。
     */


private:

    // ========================================================
    // 21. check_type<T>()
    // ========================================================

    template <typename T>
    void check_type() const {

        // static_assert：
        //
        // 编译阶段检查。
        //
        // 这里只允许：
        //
        // float
        // int32_t
        // int8_t
        //
        // 如果你写：
        //
        // tensor.ptr<double>();
        //
        // 编译直接失败。
        static_assert(
            std::is_same_v<T, float> ||
            std::is_same_v<T, int32_t> ||
            std::is_same_v<T, int8_t>,
            "Tensor supports float, int32_t, and int8_t only"
        );


        // 根据模板参数 T，
        // 计算它应该对应什么 DataType。
        //
        //
        // 如果：
        //
        // T == float
        //
        // expected = FLOAT32
        //
        //
        // 否则：
        //
        // T == int32_t
        //
        // expected = INT32
        const auto expected =
            std::is_same_v<T, float>
                ? DataType::FLOAT32
                : (std::is_same_v<T, int32_t>
                    ? DataType::INT32
                    : DataType::INT8);


        // Tensor 本身记录的数据类型，
        // 必须和用户要求访问的数据类型一样。
        //
        // 例如：
        //
        // tensor.dtype_ = FLOAT32
        //
        // tensor.ptr<int32_t>()
        //
        // expected = INT32
        //
        // FLOAT32 != INT32
        //
        // 所以抛异常。
        if (dtype_ != expected) {
            throw std::invalid_argument(
                "Tensor dtype mismatch"
            );
        }
    }


    // ========================================================
    // 22. 检查 Tensor 有没有真正的数据内存
    // ========================================================

    void require_storage() const;

    /*
     *
     *
     * 因为：
     *
     * Tensor x;
     *
     * 是允许存在的。
     *
     *
     * 但是：
     *
     * x.ptr<float>();
     *
     * 就不行。
     *
     * 因为根本没有 Buffer。
     */


    // ========================================================
    // 23. 检查数组下标
    // ========================================================

    void check_index(std::size_t offset) const;

    /*
     * 可能实现：
     *
     * if (offset >= size_) {
     *
     *     throw std::out_of_range(...);
     * }
     *
     *
     * 同时可能检查：
     *
     * Tensor 是否在 CPU。
     *
     * 因为 index() 不允许直接访问 CUDA device memory。
     */


    // ========================================================
    // 24. 根据 shape 安全计算元素数量
    // ========================================================

    static std::size_t checked_size(
        const std::vector<int32_t>& dims
    );

    /*
     * 举例：
     *
     * dims = {2, 3, 4}
     *
     * 返回：
     *
     * 2 × 3 × 4
     * =
     * 24
     *
     *
     * 为什么叫 checked_size？
     *
     * 因为不能简单粗暴：
     *
     * size *= dim;
     *
     *
     * 还需要检查：
     *
     * ① dim 是否 <= 0
     *
     * 这个 Tensor 明确规定：
     *
     * 只支持正维度。
     *
     *
     * 所以：
     *
     * {2, 0, 3}
     *
     * 不允许。
     *
     *
     * {-1, 3}
     *
     * 也不允许。
     *
     *
     * ② 是否发生整数溢出
     *
     * 如果 shape 特别大：
     *
     * a × b × c
     *
     * 超过 std::size_t 能保存的最大值，
     * 必须报错。
     */


    // ========================================================
    // 25. Tensor 的数据类型
    // ========================================================

    DataType dtype_ = DataType::UNKNOWN;

    /*
     * 例如：
     *
     * FLOAT32
     *
     * INT32
     *
     *
     * 默认：
     *
     * UNKNOWN
     */


    // ========================================================
    // 26. Tensor 的形状
    // ========================================================

    std::vector<int32_t> dims_;

    /*
     * 例如：
     *
     * shape = [2,3,4]
     *
     * 那么：
     *
     * dims_ = {2,3,4}
     */


    // ========================================================
    // 27. Tensor 的元素总数量
    // ========================================================

    std::size_t size_ = 0;

    /*
     * 例如：
     *
     * dims = [2,3,4]
     *
     * size_ =
     *
     * 2 × 3 × 4
     *
     * =
     *
     * 24
     *
     *
     * 注意不是：
     *
     * 24 字节。
     *
     * 而是：
     *
     * 24 个元素。
     */


    // ========================================================
    // 28. Tensor 真正的数据存储
    // ========================================================

    std::shared_ptr<Buffer> buffer_;

    /*
     * 这是整个 Tensor 设计里面最重要的成员。
     *
     *
     * Tensor 并不直接保存：
     *
     * float* data_;
     *
     * 而是：
     *
     * shared_ptr<Buffer>
     *
     *
     * 整体结构：
     *
     *
     * Tensor
     * ┌─────────────────────┐
     * │ dtype_ = FLOAT32    │
     * │                     │
     * │ dims_ = {2,3}       │
     * │                     │
     * │ size_ = 6           │
     * │                     │
     * │ buffer_ ────────────────┐
     * └─────────────────────┘   │
     *                           ↓
     *                    ┌──────────────┐
     *                    │    Buffer    │
     *                    │              │
     *                    │ ptr          │
     *                    │ bytes        │
     *                    │ device       │
     *                    │ allocator    │
     *                    └──────┬───────┘
     *                           │
     *                           ↓
     *
     *                    CPU / CUDA 内存
     *
     *
     *
     * shared_ptr 最大的作用：
     *
     * 自动管理 Buffer 生命周期。
     *
     *
     * 比如：
     *
     * Tensor a = ...;
     *
     * {
     *
     *     Tensor b = a;
     *
     * }
     *
     *
     * b 销毁以后，
     *
     * Buffer 不会释放。
     *
     * 因为 a 还在使用。
     *
     *
     * 只有最后一个 Tensor
     * 不再引用这个 Buffer 时，
     *
     * Buffer 才真正析构，
     * 最终释放 CPU/CUDA 内存。
     */
};

}  // namespace mini_infer
