#pragma once

#include "mini_infer/ops/mini_infer_kernels.h"
#include "mini_infer/tensor/mini_infer_tensor.h"

namespace mini_infer {

// Slots share Tensor storage. Bindings validate device/dtype; check() validates
// that all required slots are present and their shapes and ranges are compatible.
class Layer {
  public:
    virtual ~Layer() = default;
    void set_input(std::size_t i, const Tensor& value);
    void set_output(std::size_t i, const Tensor& value);
    void set_weight(std::size_t i, const Tensor& value);
    const Tensor& get_input(std::size_t i) const { return inputs_.at(i); }
    const Tensor& get_output(std::size_t i) const { return outputs_.at(i); }
    const Tensor& get_weight(std::size_t i) const { return weights_.at(i); }
    DeviceType device_type() const { return device_; }
    void forward(cudaStream_t stream = nullptr);
    virtual void check() const = 0;

  protected:
    Layer(DeviceType device, std::size_t inputs, std::size_t outputs, std::size_t weights = 0);
    void check_tensor(const Tensor& value) const;
    virtual void compute(cudaStream_t stream) = 0;
    Tensor& output(std::size_t i) { return outputs_.at(i); }

  private:
    DeviceType device_;
    std::vector<Tensor> inputs_, outputs_, weights_;
};

class AddLayer final : public Layer {
  public:
    explicit AddLayer(DeviceType device);
    void check() const override;

  private:
    void compute(cudaStream_t stream) override;
};

class RMSNormLayer final : public Layer {
  public:
    explicit RMSNormLayer(DeviceType device, float eps = 1e-5f,
                          RMSNormKernelMode mode = RMSNormKernelMode::BLOCK);
    RMSNormKernelMode kernel_mode() const { return kernel_mode_; }
    void check() const override;

  private:
    void compute(cudaStream_t stream) override;
    float eps_;
    RMSNormKernelMode kernel_mode_;
};

} // namespace mini_infer
