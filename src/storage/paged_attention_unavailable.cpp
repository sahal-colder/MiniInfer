#include "infer/storage/paged_attention.h"
#include <stdexcept>
namespace infer {
void paged_attention_f32(const float*,const float* const*,float*,std::size_t,std::size_t,
    std::size_t,int,int,int,std::size_t,cudaStream_t) {
    throw std::runtime_error("Paged attention requires CUDA and INFER_BUILD_PAGED_KERNELS=ON");
}
}

namespace infer {
void paged_attention_qk_f32(const float*,const float* const*,float*,std::size_t,std::size_t,int,int,int,int,cudaStream_t) {
    throw std::runtime_error("Paged attention requires CUDA kernels");
}
void paged_attention_av_f32(const float*,const float* const*,float*,std::size_t,std::size_t,int,int,int,int,cudaStream_t) {
    throw std::runtime_error("Paged attention requires CUDA kernels");
}
}
