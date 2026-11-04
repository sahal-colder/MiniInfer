#include "infer/storage/paged_attention.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cmath>
#include <algorithm>
#include <stdexcept>
namespace infer {
namespace {
__global__ void attend(const float* query,const float* const* pages,float* output,
                      std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim) {
    const int d=threadIdx.x, head=blockIdx.x, kv=head/(heads/kv_heads);
    __shared__ float reduction[256];
    __shared__ float maximum, denominator, alpha, beta;
    float value=0;
    if (!d) { maximum=-CUDART_INF_F; denominator=0; }
    __syncthreads();
    for (std::size_t token=0;token<context;++token) {
        // PAGED_ADDRESSING: no gather into a contiguous history or KV copy.
        const float* page=pages[token/tokens];
        const auto offset=(token%tokens*kv_heads+kv)*dim;
        reduction[d]=d<dim ? query[head*dim+d]*page[offset+d] : 0;
        __syncthreads();
        for (int stride=128;stride;stride/=2) {
            if (d<stride) reduction[d]+=reduction[d+stride];
            __syncthreads();
        }
        if (!d) {
            const float score=reduction[0]*rsqrtf(float(dim));
            const float next=fmaxf(maximum,score);
            alpha=expf(maximum-next); beta=expf(score-next);
            denominator=denominator*alpha+beta; maximum=next;
        }
        __syncthreads();
        if (d<dim) value=value*alpha+beta*page[tokens*kv_heads*dim+offset+d];
        __syncthreads();
    }
    if (d<dim) output[head*dim+d]=value/denominator;
}
}
void paged_attention_f32(const float* query,const float* const* pages,float* output,
    std::size_t count,std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim,
    std::size_t physical,cudaStream_t stream) {
    if (!query || !pages || !output || !stream || !tokens || !context || !count ||
        heads<=0 || kv_heads<=0 || heads%kv_heads || dim<=0 || dim>256 ||
        tokens>SIZE_MAX/std::size_t(kv_heads)/std::size_t(dim)/8 ||
        2*tokens*std::size_t(kv_heads)*std::size_t(dim)*sizeof(float)>physical ||
        (context-1)/tokens>=count)
        throw std::invalid_argument("Invalid paged attention geometry");
    attend<<<heads,256,0,stream>>>(query,pages,output,context,tokens,heads,kv_heads,dim);
    const auto rc=cudaGetLastError();
    if (rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(rc));
}
}
namespace infer {
namespace {
__device__ float compatible_dot(const float* q,const float* k,int dim) {
    float sum=0.0f;int i=0;
    if ((reinterpret_cast<std::uintptr_t>(q)%alignof(float4)==0) &&
        (reinterpret_cast<std::uintptr_t>(k)%alignof(float4)==0)) {
        for(;i<=dim-4;i+=4) {
            const float4 a=*reinterpret_cast<const float4*>(q+i);
            const float4 b=*reinterpret_cast<const float4*>(k+i);
            sum+=a.x*b.x+a.y*b.y+a.z*b.z+a.w*b.w;
        }
    }
    for(;i<dim;++i) sum+=q[i]*k[i];
    return sum;
}
__global__ void compatible_qk(const float* query,const float* const* pages,float* scores,
    std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim,int capacity,
    const int32_t* device_position) {
    if (device_position) context = *device_position + 1;
    const int kv_dim=kv_heads*dim,kv_mul=heads/kv_heads;
    for(std::size_t h=blockIdx.x;h<std::size_t(heads);h+=gridDim.x) {
        const float* q=query+h*dim;float* score=scores+h*capacity;
        const auto head_offset=(h/kv_mul)*dim;const float scale=rsqrtf(float(dim));
        for(std::size_t t=threadIdx.x;t<context;t+=blockDim.x) {
            const float* key=pages[t/tokens]+t%tokens*kv_dim+head_offset;
            score[t]=compatible_dot(q,key,dim)*scale;
        }
    }
}
__global__ void compatible_av(const float* scores,const float* const* pages,float* output,
    std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim,int capacity,
    const int32_t* device_position) {
    if (device_position) context = *device_position + 1;
    const int kv_dim=kv_heads*dim,kv_mul=heads/kv_heads;
    for(std::size_t h=blockIdx.x;h<std::size_t(heads);h+=gridDim.x) {
        const float* score=scores+h*capacity;const auto head_offset=(h/kv_mul)*dim;
        float* out=output+h*dim;
        for(std::size_t i=threadIdx.x;i<std::size_t(dim);i+=blockDim.x) {
            float sum=0.0f;
            for(std::size_t t=0;t<context;++t)
                sum+=score[t]*pages[t/tokens][tokens*kv_dim+t%tokens*kv_dim+head_offset+i];
            out[i]=sum;
        }
    }
}
void compatible_geometry(const void* a,const void* b,const void* c,std::size_t context,
    std::size_t tokens,int heads,int kv_heads,int dim,int capacity,cudaStream_t stream) {
    if(!a||!b||!c||!stream||!context||!tokens||heads<=0||kv_heads<=0||heads%kv_heads||dim<=0||
       capacity<=0||context>std::size_t(capacity)||kv_heads>INT32_MAX/dim||
       tokens>SIZE_MAX/std::size_t(kv_heads)/std::size_t(dim)/8||tokens*kv_heads*dim%4)
        throw std::invalid_argument("Invalid compatible paged attention geometry/alignment");
}
}
void paged_attention_qk_f32(const float* query,const float* const* pages,float* scores,
    std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim,int capacity,cudaStream_t stream,
    const int32_t* device_position) {
    compatible_geometry(query,pages,scores,context,tokens,heads,kv_heads,dim,capacity,stream);
    compatible_qk<<<std::min(heads,65535),128,0,stream>>>(query,pages,scores,context,tokens,heads,kv_heads,dim,capacity,device_position);
    const auto rc=cudaGetLastError();if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}
void paged_attention_av_f32(const float* scores,const float* const* pages,float* output,
    std::size_t context,std::size_t tokens,int heads,int kv_heads,int dim,int capacity,cudaStream_t stream,
    const int32_t* device_position) {
    compatible_geometry(scores,pages,output,context,tokens,heads,kv_heads,dim,capacity,stream);
    compatible_av<<<std::min(heads,65535),128,0,stream>>>(scores,pages,output,context,tokens,heads,kv_heads,dim,capacity,device_position);
    const auto rc=cudaGetLastError();if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));
}

namespace {
__global__ void store_kv(const float* key, const float* value, const float* const* pages,
                         std::size_t tokens, int width, const int32_t* device_position) {
    const auto position = static_cast<std::size_t>(*device_position);
    float* page = const_cast<float*>(pages[position / tokens]);
    const auto row = position % tokens * width;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < std::size_t(width); i += blockDim.x * gridDim.x) {
        page[row + i] = key[i];
        page[tokens * width + row + i] = value[i];
    }
}
}
void paged_kv_store_f32(const float* key, const float* value, const float* const* pages,
    std::size_t tokens, int width, const int32_t* device_position, cudaStream_t stream) {
    store_kv<<<std::min((width + 255) / 256, 65535), 256, 0, stream>>>(
        key, value, pages, tokens, width, device_position);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}
