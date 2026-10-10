#include <mini_infer/core/mini_infer_tensor.h>
#include <mini_infer/model/mini_infer_model.h>
#include <mini_infer/ops/matmul/mini_infer_sgemv.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        using namespace mini_infer;
        if (argc != 2) throw std::runtime_error("expected model path");
        auto cpu = std::make_shared<CPUDeviceAllocator>();
        Tensor x(DataType::FLOAT32, {3}, cpu);
        Tensor w(DataType::FLOAT32, {2, 3}, cpu);
        Tensor y(DataType::FLOAT32, {2}, cpu);
        for (int i = 0; i < 3; ++i) x.index<float>(i) = float(i + 1);
        for (int i = 0; i < 6; ++i) w.index<float>(i) = float(i + 1);
        sgemv_kernel_cpu(x, w, y, 0.5f, nullptr);
        if (y.index<float>(0) != 7.0f || y.index<float>(1) != 16.0f)
            throw std::runtime_error("installed library GEMV mismatch");

        LlamaModel model(argv[1], cpu);
        Tensor first = model.forward(1).clone(cpu);
        model.reset();
        const Tensor& second = model.forward(1);
        if (first.size() != std::size_t(model.config().vocab_size))
            throw std::runtime_error("installed model logits shape mismatch");
        for (std::size_t i = 0; i < first.size(); ++i)
            if (!std::isfinite(second.index<float>(i)) ||
                first.index<float>(i) != second.index<float>(i))
                throw std::runtime_error("installed model reset mismatch");
        std::cout << "PASS relocated CMake package: CPU GEMV and model inference\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
