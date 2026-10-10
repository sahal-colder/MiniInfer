#include "mini_infer/ops/base/mini_infer_layer.h"

// 这里声明了真正执行计算的 kernel，例如：
//
// add_kernel_cpu
// add_kernel_cuda
//
// rmsnorm_kernel_cpu
// rmsnorm_kernel_cuda
//
// Layer 负责“调度”，
// kernel 负责“真正计算”。
#include "mini_infer/backend/mini_infer_kernels.h"


// std::isfinite
//
// 用来判断一个浮点数是不是正常的有限值。
//
// 例如：
//
// 1.0       -> true
// 0.00001   -> true
// +inf      -> false
// -inf      -> false
// NaN       -> false
#include <cmath>


// std::move
//
// 用于资源转移。
//
// 例如：
//
// name_(std::move(name))
//
// 可以把 string 内部资源移动给成员变量，
// 避免额外复制。
#include <utility>


namespace mini_infer {


// ============================================================
// Layer 构造函数
// ============================================================
//
// Layer 是所有计算层的基类。
//
// 这里主要初始化：
//
// device_
//     当前 Layer 在 CPU 还是 CUDA 上运行
//
// type_
//     当前是什么类型的 Layer
//
// name_
//     Layer 名字
//
// inputs_
//     输入槽位
//
// outputs_
//     输出槽位
//
//
// 例如 AddLayer：
//
// inputs = 2
// outputs = 1
//
// 那么：
//
// inputs_  = [Tensor(), Tensor()]
//
// outputs_ = [Tensor()]
//
// 一开始这些 Tensor 都还是默认构造出来的空 Tensor。
// 后续通过：
//
// set_input()
//
// set_output()
//
// 把真正的 Tensor 放进去。
//
Layer::Layer(
    DeviceType device,
    LayerType type,
    std::string name,
    std::size_t inputs,
    std::size_t outputs
)
    : device_(device),

      // 保存 Layer 类型
      //
      // 例如：
      //
      // LayerType::ADD
      //
      // LayerType::RMSNORM
      type_(type),

      // 把传进来的字符串资源移动给 name_。
      //
      // 比如：
      //
      // name = "add"
      //
      // 最终：
      //
      // name_ = "add"
      name_(std::move(name)),

      // 创建 inputs 个默认 Tensor。
      //
      // 如果：
      //
      // inputs = 2
      //
      // 相当于：
      //
      // inputs_.size() == 2
      inputs_(inputs),

      // 创建 outputs 个默认 Tensor。
      outputs_(outputs) {

    // Layer 目前只允许：
    //
    // CPU
    //
    // 或：
    //
    // CUDA
    //
    // UNKNOWN 不允许。
    if (
        device != DeviceType::CPU &&
        device != DeviceType::CUDA
    ) {
        throw std::invalid_argument(
            "Layer needs a known device"
        );
    }
}


// ============================================================
// Layer::check_tensor
// ============================================================
//
// 对某个 Tensor 做“通用检查”。
//
// 不管是 Add 还是 RMSNorm，
// 都需要检查这些公共条件：
//
// 1. Tensor 有没有真正的内存
// 2. Tensor 和 Layer 是否在同一个设备
// 3. Tensor 是否是 FP32
//
// 子类自己的 check() 再检查：
//
// shape
// weight
// overlap
// 等额外条件。
//
void Layer::check_tensor(
    const Tensor& value
) const {

    // --------------------------------------------------------
    // 1. Tensor 必须已经有底层 Buffer
    // --------------------------------------------------------
    //
    // Tensor::empty()
    //
    // 实际检查：
    //
    // buffer_ 是否为空。
    //
    // 如果：
    //
    // Tensor t;
    //
    // 那么 t 是默认 Tensor，
    // 还没有真正存储空间。
    //
    if (value.empty()) {
        throw std::invalid_argument(
            "Layer tensor has no storage"
        );
    }


    // --------------------------------------------------------
    // 2. Tensor 和 Layer 必须在同一个设备
    // --------------------------------------------------------
    //
    // 假设：
    //
    // Layer 在 CUDA
    //
    // 但是 Tensor 在 CPU
    //
    // 那 CUDA kernel 就不能直接拿 CPU Tensor
    // 当成 CUDA Tensor 使用。
    //
    // 所以必须保持一致。
    //
    if (value.device_type() != device_) {
        throw std::invalid_argument(
            "Layer device mismatch"
        );
    }


    // --------------------------------------------------------
    // 3. 当前 Layer 只支持 FLOAT32
    // --------------------------------------------------------
    //
    // 虽然 Tensor 支持：
    //
    // FLOAT32
    // INT32
    //
    // 但是当前这些数学层：
    //
    // Add
    // RMSNorm
    //
    // 只实现了 float 版本。
    //
    if (value.data_type() != DataType::FLOAT32) {
        throw std::invalid_argument(
            "Layer requires FP32"
        );
    }
}


// ============================================================
// Layer::set_input
// ============================================================
//
// 设置第 i 个输入 Tensor。
//
// 例如 AddLayer：
//
// AddLayer add(...);
//
// add.set_input(0, a);
// add.set_input(1, b);
//
// 这时：
//
// inputs_[0] = a
// inputs_[1] = b
//
void Layer::set_input(
    std::size_t i,
    const Tensor& value
) {

    // inputs_.at(i)
    //
    // 获取第 i 个输入槽位。
    //
    // 使用 at() 而不是 []：
    //
    // 如果越界，会自动抛异常。
    //
    // auto&
    //
    // 表示拿到这个槽位本身的引用，
    // 不是复制一份 Tensor。
    //
    auto& slot = inputs_.at(i);


    // 先检查 Tensor：
    //
    // 有没有 storage
    // device 对不对
    // dtype 是不是 FP32
    //
    check_tensor(value);


    // 把 Tensor 赋值给输入槽位。
    //
    // 注意：
    //
    // Tensor 内部是：
    //
    // std::shared_ptr<Buffer>
    //
    // 所以：
    //
    // slot = value
    //
    // 不会复制整个 Tensor 数据。
    //
    // 更接近：
    //
    // input slot
    //      │
    //      └──────┐
    //             ▼
    //           Buffer
    //             ▲
    //             │
    // value ──────┘
    //
    // 它们共享底层 Buffer。
    //
    slot = value;
}


// ============================================================
// Layer::set_output
// ============================================================
//
// 和 set_input 类似，
// 但是绑定的是输出 Tensor。
//
// 例如：
//
// Tensor y(...);
//
// add.set_output(0, y);
//
// 后面 kernel 写：
//
// outputs_[0]
//
// 实际上就是写和 y 共享的那块 Buffer。
//
// 所以外部的 y 也可以直接看到计算结果。
//
void Layer::set_output(
    std::size_t i,
    const Tensor& value
) {

    // 获取第 i 个输出槽位。
    auto& slot = outputs_.at(i);


    // 输出 Tensor 同样必须：
    //
    // 有内存
    // device 一样
    // FLOAT32
    //
    check_tensor(value);


    // Tensor 普通复制共享 Buffer。
    //
    // 所以 Layer 内部写 slot 的底层数据，
    // 外面的 value 对应 Tensor 也能看到。
    //
    slot = value;
}


// ============================================================
// Layer::forward
// ============================================================
//
// 所有 Layer 的统一执行入口。
//
// 例如：
//
// add.forward();
//
// 或：
//
// add.forward(stream);
//
//
//
// 整体流程：
//
//         forward()
//             │
//             ▼
//           check()
//             │
//             ▼
//       检查输入输出是否合法
//             │
//             ▼
//        compute(stream)
//             │
//             ▼
//         CPU/CUDA kernel
//
// ============================================================
void Layer::forward(
    cudaStream_t stream
) {

    // --------------------------------------------------------
    // 第一步：检查
    // --------------------------------------------------------
    //
    // 这里的 check() 是虚函数。
    //
    // 如果当前对象是：
    //
    // AddLayer
    //
    // 那么实际调用：
    //
    // AddLayer::check()
    //
    //
    // 如果是：
    //
    // RMSNormLayer
    //
    // 实际调用：
    //
    // RMSNormLayer::check()
    //
    check();


    // --------------------------------------------------------
    // 第二步：CPU Layer 不允许 CUDA stream
    // --------------------------------------------------------
    //
    // CUDA stream 是 GPU 的执行队列概念。
    //
    // 如果当前 Layer 是 CPU：
    //
    // device_ == CPU
    //
    // 却传入一个 CUDA stream，
    // 说明调用方式不合理。
    //
    if (
        device_ == DeviceType::CPU &&
        stream
    ) {
        throw std::invalid_argument(
            "CPU layer does not accept a CUDA stream"
        );
    }


    // --------------------------------------------------------
    // 第三步：真正计算
    // --------------------------------------------------------
    //
    // compute 也是虚函数。
    //
    // AddLayer：
    //
    // AddLayer::compute()
    //
    // RMSNormLayer：
    //
    // RMSNormLayer::compute()
    //
    compute(stream);
}


// ============================================================
// ParamLayer 构造函数
// ============================================================
//
// ParamLayer 是“带权重的 Layer”。
//
// 和普通 Layer 相比，多一个：
//
// weights_
//
//
//
// 比如 RMSNorm：
//
// input  = 1
// output = 1
// weight = 1
//
// 所以：
//
// ParamLayer(..., 1, 1, 1)
//
ParamLayer::ParamLayer(
    DeviceType device,
    LayerType type,
    std::string name,
    std::size_t inputs,
    std::size_t outputs,
    std::size_t weights
)
    // 先调用父类 Layer 构造函数。
    //
    // 初始化：
    //
    // device_
    // type_
    // name_
    // inputs_
    // outputs_
    //
    : Layer(
          device,
          type,
          std::move(name),
          inputs,
          outputs
      ),

      // 创建指定数量的权重槽位。
      //
      // 例如：
      //
      // weights = 1
      //
      // weights_.size() == 1
      //
      weights_(weights) {
}


// ============================================================
// ParamLayer::set_weight
// ============================================================
//
// 设置第 i 个权重 Tensor。
//
// 例如 RMSNorm：
//
// rmsnorm.set_weight(0, weight);
//
// 那么：
//
// weights_[0] = weight
//
void ParamLayer::set_weight(
    std::size_t i,
    const Tensor& value
) {

    // 找到第 i 个权重槽位。
    auto& slot = weights_.at(i);


    // 权重 Tensor 同样要满足：
//
// storage
// device
// FP32
//
    check_tensor(value);


    // 普通 Tensor 赋值。
    //
    // 底层 Buffer 共享。
    slot = value;
}


// ============================================================
// AddLayer 构造函数
// ============================================================
//
// Add 的数学形式：
//
// y = a + b
//
//
// 所以需要：
//
// 2 个输入
// 1 个输出
//
//
// 参数：
//
// device
//
// 用来指定：
//
// CPU Add
//
// 或：
//
// CUDA Add
//
AddLayer::AddLayer(
    DeviceType device
)
    : Layer(
          device,
          LayerType::ADD,
          "add",
          2,
          1
      ) {
}


// ============================================================
// AddLayer::check
// ============================================================
//
// Add 没有广播功能。
//
// 所以要求：
//
// a.shape == b.shape == y.shape
//
//
// 例如可以：
//
// a = [2,3]
// b = [2,3]
// y = [2,3]
//
//
// 不允许：
//
// a = [2,3]
// b = [3]
//
// 即使 NumPy/PyTorch 可能支持广播，
// 这个教学版目前不支持。
//
void AddLayer::check() const {

    // 获取第 0 个输入。
    //
    // a：
    //
    // y = a + b
    //
    const auto& a = get_input(0);


    // 获取第 1 个输入。
    const auto& b = get_input(1);


    // 获取第 0 个输出。
    const auto& y = get_output(0);


    // --------------------------------------------------------
    // 通用 Tensor 检查
    // --------------------------------------------------------
    //
    // 三个 Tensor 都必须：
    //
    // 有 storage
    // device 和 Layer 一样
    // dtype == FLOAT32
    //
    check_tensor(a);
    check_tensor(b);
    check_tensor(y);


    // --------------------------------------------------------
    // shape 必须完全一样
    // --------------------------------------------------------
    //
    // 条件：
    //
    // a.dims() == b.dims()
    //
    // 并且：
    //
    // a.dims() == y.dims()
    //
    //
    // 如果任意一个不一样，就报错。
    //
    if (
        a.dims() != b.dims() ||
        a.dims() != y.dims()
    ) {
        throw std::invalid_argument(
            "Add requires equal shapes; no broadcasting"
        );
    }


    // --------------------------------------------------------
    // 输出不能和输入共享/重叠内存
    // --------------------------------------------------------
    //
    // 比如不允许：
    //
    // y 和 a 指向同一块 Buffer。
    //
    //
    // 也就是说不支持这种 inplace：
    //
    // a = a + b
    //
    //
    // 为什么？
    //
    // 因为 kernel 运行过程中：
    //
    // 一边读 a
    // 一边写 y
    //
    // 如果 y 和 a 重叠，
    // 可能修改还没有读取的数据。
    //
    if (
        y.overlaps(a) ||
        y.overlaps(b)
    ) {
        throw std::invalid_argument(
            "Add output must not overlap inputs"
        );
    }
}


// ============================================================
// AddLayer::compute
// ============================================================
//
// 真正执行：
//
// y = a + b
//
//
//
// 这句代码非常重要：
//
// get_add_kernel(device_type())
//
// 会根据 device 选择不同函数：
//
// CPU
//   ↓
// add_kernel_cpu
//
// CUDA
//   ↓
// add_kernel_cuda
//
//
// 然后后面的：
//
// (...)
//
// 就是在调用返回的那个 kernel 函数。
//
void AddLayer::compute(
    cudaStream_t stream
) {

    // 可以拆开理解成：
    //
    // AddKernel kernel =
    //     get_add_kernel(device_type());
    //
    // kernel(
    //     get_input(0),
    //     get_input(1),
    //     output(0),
    //     stream
    // );
    //
    //
    // 原代码只是把两步写成一行。
    //
    get_add_kernel(
        device_type()
    )(
        get_input(0),
        get_input(1),
        output(0),
        stream
    );
}


// ============================================================
// RMSNormLayer 构造函数
// ============================================================
//
// RMSNorm 是带权重的 Layer。
//
// 所以继承的是：
//
// ParamLayer
//
//
// RMSNorm 的结构：
//
// input:
//
// x
//
// weight:
//
// w
//
// output:
//
// y
//
//
// 所以：
//
// inputs  = 1
// outputs = 1
// weights = 1
//
RMSNormLayer::RMSNormLayer(DeviceType device, float eps)
    : RMSNormLayer(device, eps, RMSNormKernelMode::BLOCK) {}

RMSNormLayer::RMSNormLayer(
    DeviceType device,
    float eps,
    RMSNormKernelMode kernel_mode
)
    : ParamLayer(
          device,
          LayerType::RMSNORM,
          "rmsnorm",
          1,
          1,
          1
      ),

      // 保存 epsilon。
      eps_(eps),
      kernel_mode_(kernel_mode) {

    // --------------------------------------------------------
    // 检查 eps
    // --------------------------------------------------------
    //
    // std::isfinite(eps)
    //
    // 检查 eps 不是：
    //
    // NaN
    // +inf
    // -inf
    //
    //
    // 同时 eps 必须 > 0。
    //
    // 例如：
    //
    // 1e-5
    //
    // 合法。
    //
    if (
        !std::isfinite(eps) ||
        eps <= 0
    ) {
        throw std::invalid_argument(
            "RMSNorm eps must be finite and positive"
        );
    }
}


// ============================================================
// RMSNormLayer::check
// ============================================================
//
// RMSNorm shape 规则：
//
// x = [..., D]
//
// w = [D]
//
// y = [..., D]
//
//
// 例如：
//
// x = [2,3,4096]
//
// w = [4096]
//
// y = [2,3,4096]
//
//
// RMSNorm 沿最后一维 D 进行计算。
//
void RMSNormLayer::check() const {

    // 输入 x。
    const auto& x = get_input(0);


    // RMSNorm 权重。
    const auto& w = get_weight(0);


    // 输出 y。
    const auto& y = get_output(0);


    // --------------------------------------------------------
    // 通用 Tensor 检查
    // --------------------------------------------------------
    //
    // x / w / y 都必须：
    //
    // 有 storage
    // device 一样
    // dtype == FLOAT32
    //
    check_tensor(x);
    check_tensor(w);
    check_tensor(y);


    // --------------------------------------------------------
    // shape 检查
    // --------------------------------------------------------
    //
    // 第一个条件：
    //
    // x.dims() != y.dims()
    //
    // RMSNorm 输入输出 shape 必须完全一样。
    //
    //
    // 第二个条件：
    //
    // w.dims().size() != 1
    //
    // weight 必须是一维 Tensor。
    //
    // 例如：
    //
    // [4096]
    //
    // 合法。
    //
    //
    // 第三个条件：
    //
    // w.dim(0) != x.dims().back()
    //
    // weight 长度必须等于 x 最后一维。
    //
    //
    // x.dims().back()
    //
    // 就是 vector 最后一个元素。
    //
    // 比如：
    //
    // x.dims() = {2,3,4096}
    //
    // back() = 4096
    //
    if (
        x.dims() != y.dims() ||
        w.dims().size() != 1 ||
        w.dim(0) != x.dims().back()
    ) {
        throw std::invalid_argument(
            "RMSNorm expects x=[...,D], w=[D], y=shape(x)"
        );
    }


    // --------------------------------------------------------
    // 输出不能和输入/权重重叠
    // --------------------------------------------------------
    //
    // 不允许：
//
// y 和 x 共用内存
//
// 也不允许：
//
// y 和 weight 共用内存
//
    if (
        y.overlaps(x) ||
        y.overlaps(w)
    ) {
        throw std::invalid_argument(
            "RMSNorm output must not overlap input or weight"
        );
    }
}


// ============================================================
// RMSNormLayer::compute
// ============================================================
//
// 这里不直接写 RMSNorm 算法。
//
// 而是根据设备选择真正的 kernel。
//
//
// CPU：
//
// rmsnorm_kernel_cpu
//
//
// CUDA：
//
// rmsnorm_kernel_cuda
//
//
// 然后传入：
//
// x
// weight
// y
// eps
// stream
//
void RMSNormLayer::compute(
    cudaStream_t stream
) {

    // 同样可以拆开成：
    //
    // RMSNormKernel kernel =
    //     get_rmsnorm_kernel(device_type());
    //
    // kernel(
    //     get_input(0),
    //     get_weight(0),
    //     output(0),
    //     eps_,
    //     stream
    // );
    //
    get_rmsnorm_kernel(
        device_type(),
        kernel_mode_
    )(
        get_input(0),
        get_weight(0),
        output(0),
        eps_,
        stream
    );
}


// ============================================================
// get_add_kernel
// ============================================================
//
// 根据 DeviceType 返回对应的 Add kernel。
//
// 注意：
//
// 这里“返回的不是计算结果”。
//
// 返回的是：
//
// 一个函数。
//
//
// 例如：
//
// AddKernel kernel =
//     get_add_kernel(DeviceType::CPU);
//
// 得到的 kernel 实际就是：
//
// add_kernel_cpu
//
//
// 然后可以：
//
// kernel(a, b, y, stream);
//
//
//
// 这叫：
//
// function dispatch
//
// 或：
//
// backend dispatch
//
// 即根据设备选择不同后端实现。
//
AddKernel get_add_kernel(
    DeviceType device
) {

    switch (device) {

    // CPU：
    //
    // 返回 CPU 版本函数。
    case DeviceType::CPU:
        return add_kernel_cpu;


    // CUDA：
    //
    // 返回 CUDA 版本函数。
    case DeviceType::CUDA:
        return add_kernel_cuda;


    // UNKNOWN 或其他非法值：
    //
    // 不知道应该使用哪个实现。
    default:
        throw std::invalid_argument(
            "Unknown Add backend"
        );
    }
}


// ============================================================
// get_rmsnorm_kernel
// ============================================================
//
// 和 get_add_kernel 完全一样的设计。
//
// 只是这里选择的是 RMSNorm kernel。
//
RMSNormKernel get_rmsnorm_kernel(DeviceType device) {
    return get_rmsnorm_kernel(device, RMSNormKernelMode::BLOCK);
}

RMSNormKernel get_rmsnorm_kernel(
    DeviceType device,
    RMSNormKernelMode mode
) {

    switch (device) {

    // CPU RMSNorm。
    case DeviceType::CPU:
        return rmsnorm_kernel_cpu;


    // CUDA RMSNorm。
    case DeviceType::CUDA:
        switch (mode) {
        case RMSNormKernelMode::WARP:
            return rmsnorm_kernel_cuda_warp;
        case RMSNormKernelMode::BLOCK:
            return rmsnorm_kernel_cuda;
        case RMSNormKernelMode::VECTOR4:
            return rmsnorm_kernel_cuda_vectorized;
        }
        throw std::invalid_argument("Unknown RMSNorm kernel mode");


    default:
        throw std::invalid_argument(
            "Unknown RMSNorm backend"
        );
    }
}


}  // namespace mini_infer
