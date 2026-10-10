#pragma once

#include "mini_infer/ops/base/mini_infer_layer.h"
#include "mini_infer/weights/mini_infer_mmap.h"

#include <memory>
#include <vector>

namespace mini_infer {

class FP32WeightLoader {
public:
    explicit FP32WeightLoader(std::shared_ptr<const MappedModelFile> model);

    const ModelConfig& config() const noexcept { return model_->config(); }
    std::size_t position() const noexcept { return position_; }
    std::size_t remaining() const noexcept { return total_count_ - position_; }

    Tensor read_tensor(std::vector<int32_t> dims);

    void load_weight(
        ParamLayer& layer,
        std::size_t slot,
        std::vector<int32_t> dims
    );

private:
    std::shared_ptr<const MappedModelFile> model_;
    std::size_t total_count_ = 0;
    std::size_t position_ = 0;
};

}
