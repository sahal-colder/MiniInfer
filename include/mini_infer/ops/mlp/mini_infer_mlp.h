#pragma once

// 第 17 课学习顺序：
//   1. 本文件：公式、形状、工作区和所有权。
//   2. mini_infer_mlp.cpp 的 forward：课件的六步整体流程。
//   3. mini_infer_swiglu.h/.cpp/.cu：先 CPU 公式，再 GPU 线程映射。
//   4. mini_infer_swiglu_armadillo.cpp：课件 CPU 数学库写法。
//   5. mini_infer_mlp_weight_loader.h/.cpp：最后再看文件偏移。
#include "mini_infer/ops/matmul/mini_infer_matmul.h"
#include "mini_infer/ops/mlp/mini_infer_swiglu.h"

namespace mini_infer {

// 都是 FP32，三次线性变换均无 bias，矩阵统一按行优先 [out,in] 保存。
// 课件正文有两处笔误：降维用 W2，不是 W3；hidden_dim 不是固定的
// 1.5*dim。它由模型配置决定（训练代码还可能按 multiple_of 对齐）。
// 此处直接使用已确定的 hidden_dim，不自行改写模型的矩阵形状。
struct MLPWeights {
    Tensor norm;  // [dim]：FFN 的 RMSNorm 权重，不是 attention 的 norm。
    Tensor w1;    // [hidden_dim, dim]：gate，上映射后经过 SiLU。
    Tensor w2;    // [dim, hidden_dim]：down，降维回 dim。
    Tensor w3;    // [hidden_dim, dim]：up，上映射后直接参与逐元素乘法。
};

// 工作区由调用者提前分配，可以连续复用；forward 不重新分配这些 Tensor。
// public 成员让学习时可查看每一步结果。普通 Tensor 复制共享 Buffer，
// workspace 的副本不是独立工作区；若要并行运行，应分别构造。
struct MLPWorkspace {
    Tensor norm_output;  // [...,dim]
    Tensor w1_output;    // [...,hidden_dim]，随后原位覆盖为 SwiGLU 结果。
    Tensor w3_output;    // [...,hidden_dim]
    Tensor w2_output;    // [...,dim]，未加残差的纯 MLP 输出。

    MLPWorkspace(const std::vector<int32_t>& input_shape, int32_t hidden_dim,
                 std::shared_ptr<DeviceAllocator> allocator);
};

// 本类对应课件的 FFN 子块（包括前置 RMSNorm 和末尾残差）：
//   n = RMSNorm(x, norm)
//   gate = n * W1^T       up = n * W3^T
//   act = SiLU(gate) * up                  // * 为逐元素乘法
//   down = act * W2^T
//   output = x + down
// 输入/输出支持 [dim]、[seq,dim] 或 [...,dim]，按最后一维逐行计算。
// 这是 FP32 MLP 组件，不包含完整 Transformer/Attention 或模型生成。
class MLP {
public:
    // Tensor 句柄复制并共享保活权重 Buffer，不复制整份权重数据。
    // 使用 Tensor::borrow 的外部内存仍由调用者负责保活。
    explicit MLP(const MLPWeights& weights, float eps = 1e-5f);

    // output 可以独立分配，也可以与 input 完全重合（课件原位残差）。
    // 四个工作区必须各自独立，不能覆盖 input/output/任何权重。
    // 所有参数错误在第一次数值计算前检查；CUDA 运行错误不保证回滚。
    //
    // 用法：MLPWorkspace work(x.dims(), hidden_dim, allocator);
    //       MLP mlp(weights);
    //       mlp.forward(x, x, work);  // x <- x + MLP(RMSNorm(x))
    // CUDA 的六步均提交到同一 stream；返回不代表 GPU 完成。
    // 输入/输出/work/权重须活到计算结束，读取 CPU 结果前先同步。
    // 同一实例/工作区不得跨 stream 并发使用或在未完成时重新绑定。
    void forward(const Tensor& input, Tensor& output, MLPWorkspace& work,
                 cudaStream_t stream = nullptr);

private:
    MLPWeights weights_;
    int32_t dim_;
    int32_t hidden_dim_;
    RMSNormLayer norm_layer_;
    MatmulLayer w1_layer_;
    MatmulLayer w3_layer_;
    SwiGLULayer swiglu_layer_;
    MatmulLayer w2_layer_;
};
}  // namespace mini_infer
