#include <iostream>
#if !__has_include("mini_infer/ops/attention/mini_infer_attention_qkv.h") || !__has_include("mini_infer/weights/mini_infer_qkv_weight_loader.h")
int main() { std::cerr << "FAIL: lesson 16 QKV projection and weight loader are missing\n"; return 1; }
#else
#include "mini_infer/ops/attention/mini_infer_attention_qkv.h"
#include "mini_infer/weights/mini_infer_qkv_weight_loader.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
using namespace mini_infer;
auto cpu = std::make_shared<CPUDeviceAllocator>();
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
template<class F> void rejects(F f) { bool threw=false; try { f(); } catch (const std::exception&) { threw=true; } require(threw,"expected rejection"); }
void cuda_ok(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
Tensor tensor(std::vector<int32_t> dims, std::initializer_list<float> vals) {
    Tensor t(DataType::FLOAT32, dims, cpu); require(t.size()==vals.size(),"fixture shape");
    std::copy(vals.begin(),vals.end(),t.ptr<float>()); return t;
}
void values(const Tensor& t, std::initializer_list<float> want) {
    Tensor host=t.to_cpu(); require(host.size()==want.size(),"result shape");
    std::size_t i=0; for(float x:want) require(std::abs(host.index<float>(i++)-x)<1e-4f,"wrong numeric result");
}
// Literal matrices catch orientation errors, K/V swaps and accidentally recomputed history.
QKVWeights weights(bool gqa) {
    QKVWeights w;
    w.query=tensor({4,4},{1,0,0,0, 0,2,0,0, 0,0,3,0, 0,0,0,4});
    w.key=gqa ? tensor({2,4},{1,1,0,0, 0,0,1,-1}) : tensor({4,4},{1,1,0,0, 0,0,1,-1, 0,1,0,0, 1,0,0,1});
    w.value=gqa ? tensor({2,4},{1,0,1,0, 0,1,0,1}) : tensor({4,4},{1,0,1,0, 0,1,0,1, 0,0,1,0, 1,0,0,0});
    return w;
}
void projection(bool gpu, bool gqa) {
    std::shared_ptr<DeviceAllocator> alloc=cpu;
    if(gpu) alloc=std::make_shared<CUDADeviceAllocator>();
    auto w=weights(gqa);
    if(gpu) { w.query=w.query.to_cuda(alloc); w.key=w.key.to_cuda(alloc); w.value=w.value.to_cuda(alloc); }
    AttentionQKV qkv(w); KVCache cache(2,2,gqa?2:4,alloc);
    Tensor q(DataType::FLOAT32,{4},alloc);
    cudaStream_t stream=nullptr;
    if(gpu) cuda_ok(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    auto x=tensor({4},{1,2,3,4}); if(gpu) x=x.to_cuda(alloc);
    auto slots=qkv.forward(x,q,cache,1,0,stream);
    if(gpu) cuda_ok(cudaStreamSynchronize(stream));
    values(q,{1,4,9,16});
    if(gqa) { values(slots.first,{3,-1}); values(slots.second,{4,6}); }
    else { values(slots.first,{3,-1,2,5}); values(slots.second,{4,6,3,1}); }
    require(slots.first.ptr<float>()==cache.slice_kv_cache(1,0).first.ptr<float>(),"K must write into cache");
    require(slots.second.ptr<float>()==cache.slice_kv_cache(1,0).second.ptr<float>(),"V must write into cache");
    require(cache.valid_length(1)==0,"raw K must await RoPE/commit");
    // Simulate a caller postprocess between projection and commit, without claiming it is RoPE.
    Tensor corrected=tensor({gqa?2:4},gqa ? std::initializer_list<float>{30,-10} : std::initializer_list<float>{30,-10,20,50});
    slots.first.copy_from(corrected); cache.commit(1,0);
    auto x2=tensor({4},{-1,1,2,0}); if(gpu) x2=x2.to_cuda(alloc);
    qkv.forward(x2,q,cache,1,1,stream); if(gpu) cuda_ok(cudaStreamSynchronize(stream));
    cache.commit(1,1); values(q,{-1,2,6,0});
    if(gqa) { values(cache.history(1).first,{30,-10,0,2}); values(cache.history(1).second,{4,6,1,1}); }
    else { values(cache.history(1).first,{30,-10,20,50,0,2,1,-1}); values(cache.history(1).second,{4,6,3,1,1,1,2,-1}); }
    require(cache.valid_length(0)==0,"layer isolation");
    rejects([&]{qkv.forward(x,q,cache,1,2,stream);});
    rejects([&]{qkv.forward(x,q,cache,1,0,stream);});
    if(gpu) cuda_ok(cudaStreamDestroy(stream));
}
void checks() {
    auto w=weights(false); AttentionQKV qkv(w); KVCache cache(2,3,4,cpu); cache.reset(true);
    auto x=tensor({4},{1,2,3,4}); auto q=tensor({4},{-9,-9,-9,-9});
    auto bad=tensor({2,2},{1,2,3,4});
    rejects([&]{qkv.forward(bad,q,cache,0,0);}); values(q,{-9,-9,-9,-9});
    rejects([&]{qkv.forward(x,q,cache,0,1);});
    rejects([&]{qkv.forward(x,q,cache,-1,0);});
    rejects([&]{qkv.forward(x,q,cache,2,0);});
    rejects([&]{qkv.forward(x,x,cache,0,0);});
    auto key_as_q=cache.slice_kv_cache(0,0).first;
    rejects([&]{qkv.forward(x,key_as_q,cache,0,0);});
    auto weight_as_q=w.value.view(0,{4});
    rejects([&]{qkv.forward(x,weight_as_q,cache,0,0);});
    // Protect valid history even if the caller supplies another layer's K as Q.
    auto hk=tensor({4},{5,6,7,8}),hv=tensor({4},{9,10,11,12}); cache.append(1,hk,hv);
    auto history_as_q=cache.slice_kv_cache(1,0).first;
    rejects([&]{qkv.forward(x,history_as_q,cache,0,0);}); values(history_as_q,{5,6,7,8});
    auto cross=weights(false); cross.value=cache.slice_kv_cache(0,0).first; // wrong matrix rank
    rejects([&]{AttentionQKV wrong(cross);});
    auto malformed=weights(true); malformed.value=Tensor(); rejects([&]{AttentionQKV wrong(malformed);});
    KVCache narrow(2,3,2,cpu); rejects([&]{qkv.forward(x,q,narrow,0,0);});
    require(cache.valid_length(0)==0,"failed projection changed length");
    values(cache.slice_kv_cache(0,0).first,{0,0,0,0});
    values(cache.slice_kv_cache(0,0).second,{0,0,0,0});
    values(q,{-9,-9,-9,-9});
}
// Hand-assembled export blocks: 12 embedding + 8 norm + 32 Q + 16 K + 16 V.
// Distinct numbers identify both family and layer; wrong interleaving cannot pass.
struct Fixture {
    std::filesystem::path path;
    Fixture(const std::string& suffix, bool truncated=false, bool partial=false) {
        path=std::filesystem::temp_directory_path()/("lesson16-"+std::to_string(getpid())+suffix+".bin");
        ModelConfig config{4,4,2,2,1,-3,3};
        std::vector<float> data(84); std::fill_n(data.begin(),20,-999);
        for(int i=0;i<16;++i) { data[20+i]=float(100+i);data[36+i]=float(200+i); }
        for(int i=0;i<8;++i) { data[52+i]=float(300+i);data[60+i]=float(400+i);data[68+i]=float(500+i);data[76+i]=float(600+i); }
        if(truncated) data.pop_back();
        std::ofstream f(path,std::ios::binary); f.write(reinterpret_cast<const char*>(&config),sizeof(config));
        f.write(reinterpret_cast<const char*>(data.data()),data.size()*sizeof(float)); if(partial) f.put('x');
        require(bool(f),"fixture write");
    }
    ~Fixture(){std::filesystem::remove(path);}
};
void loader(bool gpu) {
    Fixture f("ok"); auto model=std::make_shared<MappedModelFile>(f.path.string());
    QKVWeightLoader reader(model); const auto& layout=reader.layout();
    require(layout.query_begin==20 && layout.key_begin==52 && layout.value_begin==68 && layout.end==84,"wrong export layout");
    std::shared_ptr<DeviceAllocator> alloc=cpu; if(gpu) alloc=std::make_shared<CUDADeviceAllocator>();
    auto a=reader.load_layer(0,alloc),b=reader.load_layer(1,alloc);
    values(a.key,{300,301,302,303,304,305,306,307}); values(a.value,{500,501,502,503,504,505,506,507});
    values(b.key,{400,401,402,403,404,405,406,407}); values(b.value,{600,601,602,603,604,605,606,607});
    values(a.query.view(0,{4}),{100,101,102,103}); values(b.query.view(0,{4}),{200,201,202,203});
    AttentionQKV qkv(b); KVCache cache(model->config(),alloc); Tensor q(DataType::FLOAT32,{4},alloc);
    auto x=tensor({4},{1,2,3,4}); if(gpu)x=x.to_cuda(alloc);
    auto slots=qkv.forward(x,q,cache,1,0); if(gpu)cuda_ok(cudaDeviceSynchronize());
    values(q,{2020,2060,2100,2140});values(slots.first,{4020,4060});values(slots.second,{6020,6060});
    rejects([&]{reader.load_layer(-1,alloc);});rejects([&]{reader.load_layer(2,alloc);});rejects([&]{reader.load_layer(0,nullptr);});
    QKVWeights owned;
    { QKVWeightLoader temporary(std::make_shared<MappedModelFile>(f.path.string())); owned=temporary.load_layer(0,cpu); }
    values(owned.key,{300,301,302,303,304,305,306,307});
    Fixture short_file("short",true),partial("partial",false,true);
    rejects([&]{QKVWeightLoader bad(std::make_shared<MappedModelFile>(short_file.path.string()));});
    rejects([&]{QKVWeightLoader bad(std::make_shared<MappedModelFile>(partial.path.string()));});
    rejects([&]{QKVWeightLoader bad(nullptr);});
    ModelConfig mha{4,4,2,2,2,3,3}; auto m=make_qkv_weight_layout(mha);
    require(m.key_begin==52&&m.value_begin==84&&m.end==116,"MHA offsets");
    for(int i=0;i<4;++i) { auto bad=mha; if(i==0)bad.vocab_size=0;if(i==1)bad.layer_num=0;if(i==2)bad.head_num=3;if(i==3)bad.kv_head_num=3; rejects([&]{make_qkv_weight_layout(bad);}); }
    auto huge=mha; huge.dim=2147483647;huge.head_num=1;huge.kv_head_num=1;huge.layer_num=2147483647;
    rejects([&]{make_qkv_weight_layout(huge);});
}
int main(int argc,char** argv) {
    try {
        bool gpu=argc>1&&std::string(argv[1])=="--cuda";
        projection(gpu,false); projection(gpu,true); loader(gpu); if(!gpu)checks();
        std::cout<<"PASS lesson16 "<<(gpu?"CUDA nonblocking stream":"CPU")<<": MHA/GQA, direct cache writes, history, export layout, failures\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
#endif
