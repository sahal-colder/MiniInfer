#pragma once

// Tensor 是 Layer 的输入、输出和权重的数据载体。
// Layer 本身并不直接管理裸内存，底层数据由 Tensor -> Buffer 管理。
#include "mini_infer/core/mini_infer_tensor.h"
#include "mini_infer/backend/mini_infer_kernels.h"

#include <string>


namespace mini_infer {


// ============================================================
// LayerType
// ============================================================
//
// 用来标记当前 Layer 属于哪一种层。
//
// 当前课程已实现：
//
// ADD
//     元素级加法层
//
// RMSNORM
//     RMSNorm 归一化层
//
// MATMUL
//     第 11 课的无 bias 全连接层，实现在 mini_infer_matmul.h/.cpp
//
// 以后还可以继续增加：
//
// LINEAR
// RELU
// ATTENTION
// ROPE
// SOFTMAX
// 等等。
//
enum class LayerType {
    ADD,
    RMSNORM,
    MATMUL
};



// ============================================================
// Layer
// ============================================================
//
// Layer 是所有“计算层”的基类。
//
// 可以把它理解为：
//
//                   Layer
//                     │
//         ┌───────────┼───────────┐
//         │           │           │
//      输入 Tensor   输出 Tensor   基本信息
//                               device
//                               type
//                               name
//
//
// Layer 自己不负责具体怎么算。
// 它主要负责：
//
// 1. 保存输入 Tensor
// 2. 保存输出 Tensor
// 3. 检查 Tensor 是否合法
// 4. 调用具体计算逻辑
//
// 真正的计算由子类实现：
//
// AddLayer::compute()
//
// RMSNormLayer::compute()
//
//
// 整体调用关系：
//
// user
//   │
//   ▼
// layer.forward()
//   │
//   ├── check()
//   │
//   └── compute()
//          │
//          ▼
//       CPU / CUDA kernel
//
// ============================================================

class Layer {

public:

    // ========================================================
    // 虚析构函数
    // ========================================================
    //
    // 因为 Layer 是一个基类，而且有虚函数，
    // 所以析构函数必须写成 virtual。
    //
    // 例如：
    //
    // Layer* layer = new AddLayer(...);
    //
    // delete layer;
    //
    // 如果基类析构函数不是 virtual，
    // 有可能只调用 Layer 的析构函数，
    // 而不调用 AddLayer 的析构函数。
    //
    // virtual 可以保证：
    //
    // delete Layer*
    //
    // 时会正确调用真正子类的析构函数。
    //
    virtual ~Layer() = default;


    // ========================================================
    // 设置第 i 个输入 Tensor
    // ========================================================
    //
    // 例如 AddLayer 有两个输入：
    //
    // input 0 = x
    // input 1 = y
    //
    // 可以：
    //
    // layer.set_input(0, x);
    // layer.set_input(1, y);
    //
    //
    // 注意这里参数是：
    //
    // const Tensor& value
    //
    // 传入时不会复制一整块数据。
    //
    // 但赋值到 inputs_ 中以后：
    //
    // Tensor 默认复制会复制 Tensor 元数据，
    // shared_ptr<Buffer> 则共享底层 Buffer。
    //
    // 所以 Layer 和外部 Tensor 通常会共享同一块底层数据。
    //
    void set_input(
        std::size_t i,
        const Tensor& value
    );


    // ========================================================
    // 设置第 i 个输出 Tensor
    // ========================================================
    //
    // 例如：
    //
    // layer.set_output(0, output_tensor);
    //
    //
    // Layer 一般不会自动随便创建输出内存，
    // 而是由外部准备好 Tensor 后绑定进来。
    //
    void set_output(
        std::size_t i,
        const Tensor& value
    );


    // ========================================================
    // 获取第 i 个输入 Tensor
    // ========================================================
    //
    // 返回：
    //
    // const Tensor&
    //
    // 表示：
    //
    // 返回原来的 Tensor 引用，
    // 不复制 Tensor，
    // 并且调用方不能通过这个接口修改 Tensor 对象。
    //
    //
    // inputs_.at(i)
    //
    // 使用 at()，
    // 如果 i 越界，会自动抛 std::out_of_range。
    //
    const Tensor& get_input(std::size_t i) const {
        return inputs_.at(i);
    }


    // ========================================================
    // 获取第 i 个输出 Tensor
    // ========================================================
    //
    // 对外提供的是 const Tensor&，
    // 所以使用 Layer 的用户只能读，
    // 不能直接替换或修改 Layer 内部保存的 Tensor 对象。
    //
    const Tensor& get_output(std::size_t i) const {
        return outputs_.at(i);
    }


    // ========================================================
    // 获取输入 Tensor 数量
    // ========================================================
    //
    // 例如 AddLayer：
    //
    // input_size() == 2
    //
    std::size_t input_size() const {
        return inputs_.size();
    }


    // ========================================================
    // 获取输出 Tensor 数量
    // ========================================================
    //
    // 例如 AddLayer：
    //
    // output_size() == 1
    //
    std::size_t output_size() const {
        return outputs_.size();
    }


    // ========================================================
    // 当前 Layer 所在设备
    // ========================================================
    //
    // 例如：
    //
    // DeviceType::CPU
    //
    // 或：
    //
    // DeviceType::CUDA
    //
    //
    // 这个值决定后面 compute() 应该调用：
    //
    // CPU kernel
    //
    // 还是：
    //
    // CUDA kernel
    //
    DeviceType device_type() const {
        return device_;
    }


    // ========================================================
    // 获取 Layer 类型
    // ========================================================
    //
    // 例如：
    //
    // LayerType::ADD
    //
    // LayerType::RMSNORM
    //
    LayerType layer_type() const {
        return type_;
    }


    // ========================================================
    // 获取 Layer 名字
    // ========================================================
    //
    // 例如：
    //
    // "Add"
    //
    // "RMSNorm"
    //
    //
    // 返回 const std::string&：
    //
    // 不复制字符串，
    // 并且调用方不能修改内部 name_。
    //
    const std::string& name() const {
        return name_;
    }


    // ========================================================
    // forward
    // ========================================================
    //
    // forward 是整个 Layer 最重要的统一入口。
    //
    // 用户一般只需要：
    //
    // layer.forward();
    //
    // 或：
    //
    // layer.forward(stream);
    //
    //
    // 内部流程一般是：
    //
    // forward()
    //   │
    //   ├── check()
    //   │      检查输入、输出、shape、dtype、device 等
    //   │
    //   └── compute(stream)
    //          真正执行计算
    //
    //
    // 为什么不让每个子类自己写 forward？
    //
    // 因为可能有人忘记：
    //
    // check();
    //
    // 直接就 compute()。
    //
    // 所以基类统一规定：
    //
    // 所有层必须：
    //
    // 先 check
    // 再 compute
    //
    //
    // cudaStream_t stream = nullptr
    //
    // 表示：
    //
    // 默认不指定 CUDA stream。
    //
    // 如果传入 stream，
    // CUDA kernel 可以提交到指定 stream。
    //
    void forward(
        cudaStream_t stream = nullptr
    );


    // ========================================================
    // check
    // ========================================================
    //
    // 纯虚函数。
    //
    // = 0 表示：
    //
    // Layer 本身不实现具体检查逻辑，
    // 每个子类必须自己实现。
    //
    //
    // AddLayer：
    //
    // 检查：
    //
    // x.shape == y.shape
    // x.dtype == y.dtype
    // output.shape == x.shape
    //
    //
    // RMSNormLayer：
    //
    // 检查：
    //
    // x 是 FP32
    // weight 是 FP32
    // weight 长度 == hidden size
    // output shape == input shape
    //
    virtual void check() const = 0;



protected:

    // ========================================================
    // Layer 构造函数
    // ========================================================
    //
    // protected 表示：
    //
    // 外部不能直接：
    //
    // Layer layer(...);
    //
    // 只能由子类调用。
    //
    //
    // 例如：
    //
    // AddLayer::AddLayer(DeviceType device)
    //
    // 内部可以：
    //
    // Layer(
    //     device,
    //     LayerType::ADD,
    //     "Add",
    //     2,
    //     1
    // )
    //
    //
    // 参数：
    //
    // device
    //     Layer 工作在哪个设备
    //
    // type
    //     Layer 类型
    //
    // name
    //     Layer 名字
    //
    // inputs
    //     有多少个输入槽位
    //
    // outputs
    //     有多少个输出槽位
    //
    Layer(
        DeviceType device,
        LayerType type,
        std::string name,
        std::size_t inputs,
        std::size_t outputs
    );


    // ========================================================
    // check_tensor
    // ========================================================
    //
    // Layer 内部通用 Tensor 检查函数。
    //
    // 子类 check() 可以复用它。
    //
    //
    // 例如可能检查：
    //
    // 1. Tensor 是否有实际存储
    // 2. Tensor device 是否和 Layer device 一样
    // 3. dtype 是否有效
    //
    //
    // 这样就不用 AddLayer、RMSNormLayer
    // 每个地方都重复写相同检查。
    //
    void check_tensor(
        const Tensor& value
    ) const;


    // ========================================================
    // compute
    // ========================================================
    //
    // 真正执行数学计算。
    //
    // 同样是纯虚函数：
    //
    // virtual ... = 0;
    //
    // 每一个具体 Layer 必须自己实现。
    //
    //
    // AddLayer：
    //
    // compute()
    //   ↓
    // y = a + b
    //
    //
    // RMSNorm：
    //
    // compute()
    //   ↓
    // RMSNorm(x, weight)
    //
    //
    // forward() 调用 compute()。
    //
    // 但是用户不能直接调用 compute()，
    // 因为 compute 是 protected。
    //
    // 这样可以保证用户必须通过：
    //
    // forward()
    //
    // 从而一定经过 check()。
    //
    virtual void compute(
        cudaStream_t stream
    ) = 0;


    // ========================================================
    // 获取可修改的输出 Tensor
    // ========================================================
    //
    // 对外：
    //
    // get_output()
    //
    // 返回：
    //
    // const Tensor&
    //
    // 不允许随便修改。
    //
    //
    // 但是 Layer 自己计算时，
    // 当然需要修改输出 Tensor 的数据。
    //
    // 所以内部提供：
    //
    // output(i)
    //
    // 返回：
    //
    // Tensor&
    //
    //
    // 例如 compute() 中：
    //
    // Tensor& y = output(0);
    //
    // 然后可以：
    //
    // y.ptr<float>()
    //
    // 写入结果。
    //
    Tensor& output(std::size_t i) {
        return outputs_.at(i);
    }



private:

    // ========================================================
    // Layer 工作设备
    // ========================================================
    //
    // CPU：
    //
    // DeviceType::CPU
    //
    // GPU：
    //
    // DeviceType::CUDA
    //
    DeviceType device_;


    // ========================================================
    // Layer 类型
    // ========================================================
    //
    // 例如：
    //
    // LayerType::ADD
    //
    LayerType type_;


    // ========================================================
    // Layer 名字
    // ========================================================
    //
    // 例如：
    //
    // "Add"
    // "RMSNorm"
    //
    std::string name_;


    // ========================================================
    // 输入 Tensor 数组
    // ========================================================
    //
    // 为什么是 vector？
    //
    // 因为不同 Layer 输入数量不同。
    //
    //
    // Add：
    //
    // inputs_[0] = a
    // inputs_[1] = b
    //
    //
    // 某些层可能只有一个输入。
    //
    std::vector<Tensor> inputs_;


    // ========================================================
    // 输出 Tensor 数组
    // ========================================================
    //
    // Add：
    //
    // outputs_[0] = y
    //
    //
    // 一个 Layer 理论上也可能有多个输出。
    //
    std::vector<Tensor> outputs_;
};



// ============================================================
// ParamLayer
// ============================================================
//
// ParamLayer 表示：
//
// “带参数 / 权重的 Layer”。
//
// Layer：
//
// 输入 + 输出
//
// ParamLayer：
//
// 输入 + 输出 + 权重
//
//
//
// 继承关系：
//
//               Layer
//              /     \ （分支）
//       AddLayer    ParamLayer
//                     │
//                     ▼
//               RMSNormLayer
//
//
// 为什么 AddLayer 不继承 ParamLayer？
//
// 因为：
//
// y = x1 + x2
//
// 不需要 weight。
//
//
// RMSNorm：
//
// y = RMSNorm(x) * weight
//
// 有 weight，
// 所以应该继承 ParamLayer。
//
// ============================================================

class ParamLayer : public Layer {

public:

    // ========================================================
    // 设置第 i 个权重 Tensor
    // ========================================================
    //
    // 例如 RMSNorm：
    //
    // layer.set_weight(0, weight);
    //
    //
    // weights_[0] 就保存 RMSNorm 的缩放权重。
    //
    void set_weight(
        std::size_t i,
        const Tensor& value
    );


    // ========================================================
    // 获取第 i 个权重 Tensor
    // ========================================================
    //
    // 返回 const Tensor&，
    // 外部只能读取。
    //
    const Tensor& get_weight(std::size_t i) const {
        return weights_.at(i);
    }


    // ========================================================
    // 获取权重数量
    // ========================================================
    //
    // RMSNorm：
    //
    // weight_size() == 1
    //
    std::size_t weight_size() const {
        return weights_.size();
    }



protected:

    // ========================================================
    // ParamLayer 构造函数
    // ========================================================
    //
    // 和 Layer 构造函数相比，
    // 多了一个：
    //
    // weights
    //
    // 表示需要几个参数 Tensor。
    //
    //
    // 例如 RMSNorm：
    //
    // inputs  = 1
    // outputs = 1
    // weights = 1
    //
    ParamLayer(
        DeviceType device,
        LayerType type,
        std::string name,
        std::size_t inputs,
        std::size_t outputs,
        std::size_t weights
    );



private:

    // ========================================================
    // 权重 Tensor 数组
    // ========================================================
    //
    // RMSNorm：
    //
    // weights_[0]
    //
    // 就是：
    //
    // gamma / weight
    //
    std::vector<Tensor> weights_;
};



// ============================================================
// AddLayer
// ============================================================
//
// AddLayer 实现元素级加法：
//
// y = x1 + x2
//
//
//
// 一般结构：
//
// input[0] ───┐
//             ├── Add ──> output[0]
// input[1] ───┘
//
//
//
// 例如：
//
// x1 = [1, 2, 3]
//
// x2 = [4, 5, 6]
//
// 输出：
//
// y = [5, 7, 9]
//
//
//
// final：
//
// 表示禁止别人再继承 AddLayer。
//
// ============================================================

class AddLayer final : public Layer {

public:

    // ========================================================
    // AddLayer 构造函数
    // ========================================================
    //
    // explicit：
    //
    // 禁止隐式类型转换。
    //
    //
    // 比如没有 explicit 时，
    // 某些场景可能把 DeviceType 自动转换成 AddLayer。
    //
    // explicit 可以避免这种不必要的隐式构造。
    //
    explicit AddLayer(
        DeviceType device
    );


    // ========================================================
    // AddLayer 参数检查
    // ========================================================
    //
    // override：
    //
    // 表示这个函数是在重写：
    //
    // Layer::check()
    //
    //
    // 编译器会帮忙检查：
    //
    // 函数签名是不是和父类一致。
    //
    //
    // 一般会检查：
    //
    // input 0 有效
    // input 1 有效
    // output 0 有效
    //
    // 三者 dtype 一致
    // 三者 shape 一致
    // 三者 device 一致
    //
    void check() const override;



private:

    // ========================================================
    // AddLayer 真正计算
    // ========================================================
    //
    // CPU：
    //
    // for (...)
    //     y[i] = a[i] + b[i];
    //
    //
    // CUDA：
    //
    // 启动 add kernel。
    //
    //
    // 用户不会直接调用 compute()。
    //
    // 用户调用：
    //
    // forward()
    //
    // forward()：
    //
    // check()
    //   ↓
    // compute()
    //
    void compute(
        cudaStream_t stream
    ) override;
};



// ============================================================
// RMSNormLayer
// ============================================================
//
// RMSNorm：
//
// Root Mean Square Layer Normalization
//
// 是大模型里面非常常见的一种归一化。
//
// 例如 LLaMA 系列模型中大量使用 RMSNorm。
//
//
// 一个简化公式：
//
//              x_i
// y_i = ------------------- * weight_i
//        sqrt(mean(x²)+eps)
//
//
// 假设：
//
// x = [x1, x2, x3, ..., x_hidden]
//
// 第一步：
//
// mean_square
//
// =
//
// (x1² + x2² + ... + x_hidden²)
//
// / hidden
//
//
// 第二步：
//
// rms
//
// =
//
// sqrt(mean_square + eps)
//
//
// 第三步：
//
// y_i
//
// =
//
// x_i / rms * weight_i
//
//
// ============================================================
//
// shape 约定：
//
// x:
//
// [..., hidden]
//
//
// weight:
//
// [hidden]
//
//
// output:
//
// 和 x 完全相同。
//
//
// 例如：
//
// x:
//
// [2, 3, 4096]
//
//
// weight:
//
// [4096]
//
//
// output:
//
// [2, 3, 4096]
//
//
// 意思是：
//
// 每一组最后维度 4096 个元素
// 单独进行一次 RMSNorm。
//
// ============================================================

class RMSNormLayer final : public ParamLayer {

public:

    // ========================================================
    // RMSNormLayer 构造函数
    // ========================================================
    //
    // 参数：
    //
    // device
    //
    //     CPU / CUDA
    //
    //
    // eps
    //
    //     为防止除 0，
    //     在平方均值里面加一个很小的数。
    //
    //
    // 默认：
    //
    // eps = 1e-5
    //
    explicit RMSNormLayer(
        DeviceType device,
        float eps = 1e-5f
    );

    // 带实现选择的课程 6/8 构造函数。单独保留两参数重载，
    // 使前面课程的默认调用方式保持不变。
    RMSNormLayer(
        DeviceType device,
        float eps,
        RMSNormKernelMode kernel_mode
    );

    // 返回当前 CUDA 实现，便于课程 6/8 的基线和优化版本对比。
    // CPU 上该选项不改变数学结果，仍使用同一个 CPU kernel。
    RMSNormKernelMode kernel_mode() const {
        return kernel_mode_;
    }


    // ========================================================
    // RMSNorm 参数检查
    // ========================================================
    //
    // 一般检查：
    //
    // input 数量正确
    // output 数量正确
    // weight 数量正确
    //
    // dtype == FLOAT32
    //
    // x、output shape 一样
    //
    // weight 是 1 维
    //
    // weight.size() == x 最后一维
    //
    // device 一致
    //
    // output 不能和 input / weight 内存重叠
    //
    void check() const override;



private:

    // ========================================================
    // RMSNorm 真正计算
    // ========================================================
    //
    // CPU 版本可能类似：
    //
    // hidden = x 最后一维
    //
    // 对每一行：
    //
    //     sum = 0
    //
    //     for j:
    //         sum += x[j] * x[j]
    //
    //     mean = sum / hidden
    //
    //     scale = 1 / sqrt(mean + eps_)
    //
    //     for j:
    //         y[j] = x[j] * scale * weight[j]
    //
    //
    // CUDA 版本则启动 RMSNorm kernel。
    //
    void compute(
        cudaStream_t stream
    ) override;


    // ========================================================
    // RMSNorm 的 epsilon
    // ========================================================
    //
    // 用来防止：
    //
    // sqrt(0)
    //
    // 或数值非常小时产生不稳定。
    //
    //
    // 一般：
    //
    // 1e-5
    // 1e-6
    //
    // 等。
    //
    float eps_;

    RMSNormKernelMode kernel_mode_;
};


}  // namespace mini_infer
