#include "mini_infer/backend/mini_infer_kernels.h"
#include <algorithm>
#include <cstdint>

namespace mini_infer {
namespace {



// XOR 归约让所有 lane 都得到总和；调用处保证完整 32 个 lane 参与。
// 注意：这与只保证 lane 0 有有效结果的 CUB WarpReduce 接口不同。
__device__ float warp_sum(float value) {
    // EXERCISE_BEGIN: warp_sum
    for (int mask = 16; mask > 0; mask >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, mask);
    return value;
    // EXERCISE_END: warp_sum
}

// 入门版本：一个 block 恰好一个 warp，一个 block 负责一行。
// D<32 时，没有输入的 lane 仍然携带 0 参加 shuffle，不能提前 return。
__global__ void rmsnorm_warp_f32(const float* x, const float* w, float* y,
                                std::size_t rows, std::size_t hidden, float eps) {
    const int lane = threadIdx.x;
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto offset = row * hidden;
        float sum = 0.0f;
        for (std::size_t j = lane; j < hidden; j += 32) {
            const float v = x[offset + j];
            sum += v * v;
        }
        sum = warp_sum(sum);
        const float scale = rsqrtf(sum / float(hidden) + eps);
        for (std::size_t j = lane; j < hidden; j += 32)
            y[offset + j] = x[offset + j] * scale * w[j];
    }
}

// 进阶版本：256线程/8个warp协作一行。Layer 默认调这个版本。
// 教学重点：线程局部和 -> warp和 -> block和 -> 共享scale -> 写回。
__global__ void rmsnorm_block_f32(const float* x, const float* w, float* y,
                                 std::size_t rows, std::size_t hidden, float eps) {
    constexpr int threads = 256;
    constexpr int warps = threads / 32;
    __shared__ float warp_sums[warps];
    __shared__ float scale;
    const int tid = threadIdx.x;
    const int lane = tid % 32;
    const int warp = tid / 32;

    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const auto offset = row * hidden;
        float sum = 0.0f;
        // 用 tid / blockDim.x 划分整行，避免不同 warp 重复处理同一列。
        for (std::size_t j = tid; j < hidden; j += blockDim.x) {
            const float v = x[offset + j];
            sum += v * v;
        }
        sum = warp_sum(sum);
        if (lane == 0) warp_sums[warp] = sum;
        __syncthreads();  // 跨 warp 读取之前，保证8个和都已经写入。

        if (warp == 0) {
            float total = lane < warps ? warp_sums[lane] : 0.0f;
            total = warp_sum(total);
            if (lane == 0) scale = rsqrtf(total / float(hidden) + eps);
        }
        __syncthreads();  // 所有线程读取 scale 之前，等待 warp 0 发布结果。
        for (std::size_t j = tid; j < hidden; j += blockDim.x)
            y[offset + j] = x[offset + j] * scale * w[j];
        __syncthreads();  // grid-stride 下一行覆盖 scale 前，本行须读完。
    }
}

// 第 8 课的 float4 版本。
//
// 访存优化不能改变第 6 课已经验证过的线程分工：
// 每个线程仍然按 tid、tid + blockDim.x、... 处理自己的列，
// 先做局部平方和，再经过“warp -> shared -> warp”两级归约。
// 这里只把连续的四个 float 打包读取/写回；最后不足四个元素的尾部
// 仍使用标量路径。若某一行的首地址没有 16 字节对齐，则该行全部
// 回退到标量路径，避免未对齐 float4 解引用带来的未定义行为。
__global__ void rmsnorm_vectorized_f32(const float* x, const float* w, float* y,
                                      std::size_t rows, std::size_t hidden,
                                      float eps) {
    // 线程块配置：固定使用 256 线程（8 个 Warp），平衡寄存器占用与 SM 占用率 (Occupancy)
    constexpr int threads = 256;
    constexpr int warps = threads / 32;

    // warp_sums: 暂存每个 Warp 归约出的局部平方和，用于跨 Warp 间归约
    __shared__ float warp_sums[warps];
    // scale: 广播整行最终计算出的归一化缩放系数 1 / sqrt(mean(x^2) + eps)
    __shared__ float scale;

    // 线程索引映射
    const int tid = threadIdx.x;        // Block 内全局线程 ID [0, 255]
    const int lane = tid % 32;          // Warp 内 Lane ID [0, 31]
    const int warp = tid / 32;          // 所属 Warp 编号 [0, 7]

    // Grid-Stride Loop: 允许启动少于行数的 Block 轮询分担所有行（处理 rows > gridDim.x 的情况）
    for (std::size_t row = blockIdx.x; row < rows; row += gridDim.x) {
        // 当前行在全局显存中的首地址偏移
        const std::size_t offset = row * hidden;
        const float* row_x = x + offset;
        float* row_y = y + offset;

        // 【对齐检查】CUDA 的 128 位向量指令 (LDG.E.128/STG.E.128) 强制要求首地址 16 字节对齐。
        // 需同时检查输入 row_x、输出 row_y 和权重 w；只要有一个未对齐，就回退到安全标量路径。
        const bool aligned =
            (reinterpret_cast<std::uintptr_t>(row_x) % alignof(float4) == 0) &&
            (reinterpret_cast<std::uintptr_t>(row_y) % alignof(float4) == 0) &&
            (reinterpret_cast<std::uintptr_t>(w) % alignof(float4) == 0);

        float sum = 0.0f;                       // 线程局部的平方和累加器
        const std::size_t pack_num = hidden / 4; // 可使用 float4 (打包 4 个 float) 访问的包数量

        if (aligned) {
            // 将指针重构为 float4*，一次加载 16 字节（合并访存宽度最大化）
            const auto* x4 = reinterpret_cast<const float4*>(row_x);
            // 线程步长循环：各线程跨 blockDim.x 个 pack 获取数据，累加自身局部的平方和
            for (std::size_t pack = tid; pack < pack_num; pack += blockDim.x) {
                const float4 value = x4[pack];
                sum += value.x * value.x;
                sum += value.y * value.y;
                sum += value.z * value.z;
                sum += value.w * value.w;
            }
        } else {
            // 【未对齐回退路径】使用传统标量循环逐步求和，避免非对齐访存引发未定义行为或硬件陷阱
            for (std::size_t j = tid; j < hidden; j += blockDim.x) {
                const float value = row_x[j];
                sum += value * value;
            }
        }

        // 【尾部余量处理】如果 hidden 不是 4 的整数倍（例如 4098），剩余的元素无法打成 float4。
        // 尾部元素（从 pack_offset 开始）统一由各线程以标量方式补算平方和。
        const std::size_t pack_offset = pack_num * 4;
        if (aligned) {
            for (std::size_t j = pack_offset + tid; j < hidden;
                 j += blockDim.x) {
                const float value = row_x[j];
                sum += value * value;
            }
        }

        // ==================== 第一级归约：Warp 内部归约 ====================
        // 每个 Warp 内的 32 个线程通过寄存器 shuffle (XOR) 完成求和，耗时极低
        sum = warp_sum(sum);
        // 各 Warp 仅由 lane 0 将该 Warp 的局部总和写入 Shared Memory，避免写冲突
        if (lane == 0) warp_sums[warp] = sum;
        // 同步屏障：确保 8 个 Warp 的 lane 0 均已完成 Shared Memory 写入
        __syncthreads();

        // ==================== 第二级归约：跨 Warp 汇聚与 Scale 计算 ====================
        // 仅由 Warp 0 负责归约所有 Warp 的和，计算全行 RMS scale
        if (warp == 0) {
            // 前 8 个 lane 加载对应 warp 的和，其余 lane 填充 0 避免干扰 shuffle 归约
            float total = lane < warps ? warp_sums[lane] : 0.0f;
            total = warp_sum(total); // Warp 0 内部再次执行 32-lane shuffle
            // Warp 0 的 lane 0 计算倒数平方根并发布到共享变量 scale: 1 / sqrt(mean(x^2) + eps)
            if (lane == 0) scale = rsqrtf(total / float(hidden) + eps);
        }
        // 同步屏障：等待 Warp 0 计算并发布 scale，确保所有线程之后读到正确的系数
        __syncthreads();

        // ==================== 第二阶段：归一化计算与向量化写回 ====================
        if (aligned) {
            const auto* x4 = reinterpret_cast<const float4*>(row_x);
            const auto* w4 = reinterpret_cast<const float4*>(w);
            auto* y4 = reinterpret_cast<float4*>(row_y);

            // 主体部分：以 128 位宽（float4）同步读取 x、w，计算后写入 y
            for (std::size_t pack = tid; pack < pack_num; pack += blockDim.x) {
                const float4 xv = x4[pack];
                const float4 wv = w4[pack];
                y4[pack] = make_float4(
                    xv.x * wv.x * scale,
                    xv.y * wv.y * scale,
                    xv.z * wv.z * scale,
                    xv.w * wv.w * scale);
            }
            // 尾部非 4 对齐的标量写回
            for (std::size_t j = pack_offset + tid; j < hidden;
                 j += blockDim.x) {
                row_y[j] = row_x[j] * w[j] * scale;
            }
        } else {
            // 未对齐回退路径的标量写回
            for (std::size_t j = tid; j < hidden; j += blockDim.x)
                row_y[j] = row_x[j] * w[j] * scale;
        }

        // 同步屏障：防止快速线程在当前行写回未完成时进入下一轮 Grid-Stride 循环，
        // 提前覆盖 shared 变量（warp_sums 和 scale），造成数据竞争 (RAW Hazard)。
        __syncthreads();
    }
}

unsigned int row_blocks(std::size_t rows) {
    return static_cast<unsigned int>(std::min<std::size_t>(rows, 65535));
}
}  // namespace

void rmsnorm_kernel_cuda_warp(const Tensor& x, const Tensor& w, Tensor& y,
                              float eps, cudaStream_t stream) {
    const std::size_t hidden = std::size_t(x.dims().back());
    const std::size_t rows = x.size() / hidden;
    rmsnorm_warp_f32<<<row_blocks(rows), 32, 0, stream>>>(
        x.ptr<float>(), w.ptr<float>(), y.ptr<float>(), rows, hidden, eps);
    check_cuda(cudaGetLastError(), "RMSNorm warp kernel launch");
}

void rmsnorm_kernel_cuda(const Tensor& x, const Tensor& w, Tensor& y,
                         float eps, cudaStream_t stream) {
    const std::size_t hidden = std::size_t(x.dims().back());
    const std::size_t rows = x.size() / hidden;
    rmsnorm_block_f32<<<row_blocks(rows), 256, 0, stream>>>(
        x.ptr<float>(), w.ptr<float>(), y.ptr<float>(), rows, hidden, eps);
    // 启动检查不能捕获所有异步执行错误；测试/调用方在同步点再检查。
    check_cuda(cudaGetLastError(), "RMSNorm block kernel launch");
}

void rmsnorm_kernel_cuda_vectorized(const Tensor& x, const Tensor& w, Tensor& y,
                                    float eps, cudaStream_t stream) {
    const std::size_t hidden = std::size_t(x.dims().back());
    const std::size_t rows = x.size() / hidden;
    rmsnorm_vectorized_f32<<<row_blocks(rows), 256, 0, stream>>>(
        x.ptr<float>(), w.ptr<float>(), y.ptr<float>(), rows, hidden, eps);
    check_cuda(cudaGetLastError(), "RMSNorm vectorized kernel launch");
}

}  // namespace mini_infer
