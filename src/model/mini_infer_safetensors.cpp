#include "mini_infer/model/mini_infer_model_weights.h"
#include "infer/io/weight_io.h"
#include "mini_infer/loader/mini_infer_qkv_weight_loader.h"
#include "mini_infer/loader/mini_infer_mlp_weight_loader.h"
#include <algorithm>
#include <charconv>
#include <stdexcept>
namespace mini_infer {
ModelWeights load_safetensors_weights(const std::string& path,std::shared_ptr<DeviceAllocator> allocator) {
    if (!allocator || allocator->device_type()!=DeviceType::CUDA)
        throw std::invalid_argument("Direct Safetensors model loading requires a CUDA allocator");
    infer::io::Safetensors file(path);
    if (file.metadata("miniinfer.layout")!="llama2c-f32-v0")
        throw std::invalid_argument("Expected MiniInfer FP32/interleaved-RoPE Safetensors layout");
    const auto number=[&](const char* key) {
        const auto text=file.metadata(key);int32_t value=0;
        const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
        if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+text.size() || value<=0)
            throw std::invalid_argument(std::string("Invalid model metadata: ")+key);
        return value;
    };
    ModelWeights result;
    auto& c=result.config;
    c.dim=number("miniinfer.dim");c.hidden_dim=number("miniinfer.hidden_dim");
    c.layer_num=number("miniinfer.layer_num");c.head_num=number("miniinfer.head_num");
    c.kv_head_num=number("miniinfer.kv_head_num");c.vocab_size=number("miniinfer.vocab_size");
    c.seq_len=number("miniinfer.seq_len");
    (void)make_qkv_weight_layout(c);(void)make_mlp_weight_layout(c);
    if ((c.dim/c.head_num)%2) throw std::invalid_argument("RoPE head size must be even");
    const bool output=std::any_of(file.tensors().begin(),file.tensors().end(),[](const auto& t){return t.name=="output.weight";});
    if (file.tensors().size()!=2+std::size_t(c.layer_num)*9+output)
        throw std::invalid_argument("Unexpected MiniInfer tensor set");
    struct Binding { Tensor* destination;const infer::io::TensorInfo* source;std::vector<int32_t> shape; };
    std::vector<Binding> bindings;
    const auto bind=[&](Tensor& tensor,const std::string& name,std::vector<int32_t> shape) {
        const auto& metadata=file.tensor(name);
        if (metadata.dtype!="F32" || metadata.shape.size()!=shape.size() ||
            !std::equal(shape.begin(),shape.end(),metadata.shape.begin()))
            throw std::invalid_argument("Model shape/dtype mismatch: "+name);
        bindings.push_back({&tensor,&metadata,std::move(shape)});
    };
    bind(result.embedding,"tok_embeddings.weight",{c.vocab_size,c.dim});
    result.layers.resize(c.layer_num);
    const auto kv_dim=c.kv_head_num*(c.dim/c.head_num);
    for (int32_t i=0;i<c.layer_num;++i) {
        auto& layer=result.layers[i];const auto prefix="layers."+std::to_string(i)+".";
        bind(layer.attention_norm,prefix+"attention_norm.weight",{c.dim});
        bind(layer.qkv.query,prefix+"attention.wq.weight",{c.dim,c.dim});
        bind(layer.qkv.key,prefix+"attention.wk.weight",{kv_dim,c.dim});
        bind(layer.qkv.value,prefix+"attention.wv.weight",{kv_dim,c.dim});
        bind(layer.output,prefix+"attention.wo.weight",{c.dim,c.dim});
        bind(layer.mlp.norm,prefix+"ffn_norm.weight",{c.dim});
        bind(layer.mlp.w1,prefix+"feed_forward.w1.weight",{c.hidden_dim,c.dim});
        bind(layer.mlp.w2,prefix+"feed_forward.w2.weight",{c.dim,c.hidden_dim});
        bind(layer.mlp.w3,prefix+"feed_forward.w3.weight",{c.hidden_dim,c.dim});
    }
    bind(result.final_norm,"norm.weight",{c.dim});
    if (output) bind(result.classifier,"output.weight",{c.vocab_size,c.dim});
    // Validation above completes before the first device allocation.
    std::vector<infer::io::GpuRead> requests;
    for(auto& binding:bindings) {
        *binding.destination=Tensor(DataType::FLOAT32,std::move(binding.shape),allocator);
        requests.push_back({binding.source->file_offset,binding.destination->ptr<float>(),binding.source->bytes});
    }
    int device=0;const auto device_rc=cudaGetDevice(&device);
    if(device_rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(device_rc));
    cudaStream_t stream=nullptr;
    const auto stream_rc=cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking);
    if(stream_rc!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(stream_rc));
    struct Stream {cudaStream_t value;~Stream(){cudaStreamDestroy(value);}} owner{stream};
    infer::io::WeightIOConfig config;config.device=device;
    auto backend=infer::io::make_weight_backend(config);
    result.loading=backend->read(file,std::move(requests),stream);
    if(!output) result.classifier=result.embedding;
    return result;
}
}
