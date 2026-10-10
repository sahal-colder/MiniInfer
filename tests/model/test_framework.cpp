#include "mini_infer/generation/mini_infer_generation.h"
#include "mini_infer/ops/rope/mini_infer_rope.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>

using namespace mini_infer;
int assertions=0;
void check(bool value,const std::string& message) {
    ++assertions;
    if(!value) throw std::runtime_error(message);
}
template<class F> void rejects(F fn,const std::string& label) {
    bool caught=false;
    try { fn(); } catch(const std::exception&) { caught=true; }
    check(caught,label);
}
std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f) throw std::runtime_error("missing reference");
    auto size=f.tellg();
    std::vector<float> data(static_cast<std::size_t>(size)/4);
    f.seekg(0); f.read(reinterpret_cast<char*>(data.data()),size);
    return data;
}
void near(const Tensor& actual,const float* expected,std::size_t count,float tol) {
    for(std::size_t i=0;i<count;++i)
        check(std::isfinite(actual.ptr<float>()[i]) &&
              std::abs(actual.ptr<float>()[i]-expected[i])<tol*(1+std::abs(expected[i])),
              "numeric mismatch at "+std::to_string(i));
}
int main(int argc,char** argv) {
    try {
        const std::string root=argv[1];
        const bool gpu=argc>2;
        auto cpu=std::make_shared<CPUDeviceAllocator>();
        std::shared_ptr<DeviceAllocator> allocator=cpu;
        if(gpu) allocator=std::make_shared<CUDADeviceAllocator>();
        static_assert(!std::is_copy_constructible<LlamaModel>::value);

        // RoPE: analytic rotation on non-power-of-two head width, GQA, custom stream.
        Tensor q(DataType::FLOAT32,{24},cpu),k(DataType::FLOAT32,{12},cpu);
        for(int i=0;i<24;++i) q.ptr<float>()[i]=.1f*(i-7);
        for(int i=0;i<12;++i) k.ptr<float>()[i]=.2f*(i+1);
        auto expected_q=q.clone(cpu),expected_k=k.clone(cpu);
        for(Tensor* x:{&expected_q,&expected_k})
            for(std::size_t i=0;i<x->size();i+=2) {
                const double angle=3.0/std::pow(10000.0,double(i%6)/6.0);
                const double a=x->ptr<float>()[i],b=x->ptr<float>()[i+1];
                x->ptr<float>()[i]=a*std::cos(angle)-b*std::sin(angle);
                x->ptr<float>()[i+1]=a*std::sin(angle)+b*std::cos(angle);
            }
        if(gpu) {q=q.to_cuda(allocator);k=k.to_cuda(allocator);}
        cudaStream_t stream=nullptr;
        if(gpu && cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess)
            throw std::runtime_error("stream create failed");
        rope_inplace(q,k,6,3,10000,stream);
        if(gpu) {check(cudaStreamSynchronize(stream)==cudaSuccess,"rope stream");cudaStreamDestroy(stream);}
        auto qh=q.to_cpu(),kh=k.to_cpu();
        near(qh,expected_q.ptr<float>(),24,2e-6f);near(kh,expected_k.ptr<float>(),12,2e-6f);
        rope_inplace(q,k,6,0);
        if(gpu) check(cudaDeviceSynchronize()==cudaSuccess,"rope zero sync");
        near(q.to_cpu(),qh.ptr<float>(),24,0.000001f);
        rejects([&]{rope_inplace(q,k,3,0);},"odd RoPE head");
        rejects([&]{rope_inplace(q,q,6,0);},"RoPE alias");
        rejects([&]{rope_inplace(q,k,6,-1);},"negative RoPE position");
        rejects([&]{rope_inplace(q,k,6,0,std::numeric_limits<float>::quiet_NaN());},"RoPE nan");

        // Full model logits versus an independent FP64, batched, causal NumPy model.
        const int32_t tokens[]={1,41,299,8,10,71,2,78};
        for(int kv:{4,2,1}) for(const std::string tied:{"tied","untied"}) {
            const std::string stem=root+"/kv"+std::to_string(kv)+"_"+tied;
            auto weights=load_model_weights(stem+".bin",allocator);
            check(weights.embedding.overlaps(weights.classifier)==(tied=="tied"),"classifier sharing");
            auto invalid_config=weights.config;invalid_config.head_num=0;
            rejects([&]{TransformerBlock bad(weights.layers[0],invalid_config);},"invalid block heads");
            invalid_config=weights.config;invalid_config.hidden_dim+=1;
            rejects([&]{TransformerBlock bad(weights.layers[0],invalid_config);},"block/config mismatch");
            LlamaModel model(stem+".bin",allocator);
            auto expected=read_floats(stem+".expected");
            check(expected.size()==2400,"reference length");
            rejects([&]{model.forward(-1);},"negative id");
            rejects([&]{model.forward(300);},"high id");
            check(model.position()==0,"invalid token preserves position");
            for(int pos=0;pos<8;++pos) {
                const auto& logits=model.forward(tokens[pos]);
                near(logits,expected.data()+pos*300,300,gpu?1e-4f:3e-5f);
                check(model.position()==pos+1,"position increments");
            }
            rejects([&]{model.forward(1);},"context overflow");
            check(model.position()==8,"overflow preserves state");
            model.reset();check(model.position()==0,"reset position");
            near(model.forward(tokens[0]),expected.data(),300,gpu?1e-4f:3e-5f);
            model.reset();
            model.forward(77);model.forward(33);model.reset();
            near(model.forward(tokens[0]),expected.data(),300,gpu?1e-4f:3e-5f);
        }
        for(const std::string name:{"truncated","trailing","bad_vocab","bad_heads","bad_seq","bad_rope","bad_size"})
            rejects([&]{load_model_weights(root+"/"+name+".bin",allocator);},"bad checkpoint "+name);

        Tokenizer tokenizer(root+"/tokenizer.bin",300);
        check(tokenizer.encode("")==std::vector<int32_t>{1},"empty prompt BOS");
        check(tokenizer.encode("hello")==std::vector<int32_t>({1,271}),"BPE score merge");
        check(tokenizer.encode("你好")==std::vector<int32_t>({1,275}),"UTF8 merge");
        for(const std::string text:{"hello","  hello","你好","🙂","a\nb"}) {
            auto ids=tokenizer.encode(text);std::string decoded;
            for(std::size_t i=1;i<ids.size();++i) decoded+=tokenizer.decode(ids[i-1],ids[i]);
            check(decoded==text,"tokenizer roundtrip");
        }
        rejects([&]{Tokenizer bad(root+"/tokenizer_short.bin",300);},"truncated tokenizer");
        rejects([&]{Tokenizer bad(root+"/tokenizer_bad_length.bin",300);},"bad tokenizer length");
        rejects([&]{Tokenizer bad(root+"/tokenizer.bin",299);},"wrong vocab");
        rejects([&]{tokenizer.decode(1,300);},"decode range");
        check(tokenizer.decode(1,2).empty(),"EOS decode");

        Tensor logits(DataType::FLOAT32,{4},cpu);
        const float scores[]={-1000,3,3,0};std::copy(scores,scores+4,logits.ptr<float>());
        Sampler greedy;check(greedy.sample(logits)==1,"greedy tie leftmost");
        Sampler top1(1,1);check(top1.sample(logits)==1,"top1");
        Sampler a(.7f,2,123),b(.7f,2,123);
        for(int i=0;i<100;++i) {
            auto id=a.sample(logits);check(id==1 || id==2,"topk excludes others");
            check(id==b.sample(logits),"seed determinism");
        }
        Sampler tiny(std::numeric_limits<float>::denorm_min(),0);
        const auto tiny_id=tiny.sample(logits);
        check(tiny_id==1 || tiny_id==2,"tiny temperature");
        rejects([&]{Sampler bad(-1);},"negative temperature");
        rejects([&]{Sampler bad(std::numeric_limits<float>::infinity());},"inf temperature");
        logits.ptr<float>()[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{greedy.sample(logits);},"nan logits");

        for(const std::string name:{"eos","bos"}) {
            LlamaModel model(root+"/"+name+".bin",allocator);
            auto result=generate(model,tokenizer,greedy,"",4);
            check(result.stop_reason==name && result.generated_tokens.empty(),"stop on "+name);
        }
        LlamaModel repeat(root+"/repeat_a.bin",allocator);
        std::string output;
        auto result=generate(repeat,tokenizer,greedy,"",3,[&](const std::string& p){output+=p;});
        check(output=="aaa" && result.generated_tokens.size()==3 && result.stop_reason=="steps","generation budget");
        result=generate(repeat,tokenizer,greedy,"",100);
        check(result.generated_tokens.size()==8 && result.stop_reason=="context","context stop");
        check(repeat.position()==8,"last context logits used");
        result=generate(repeat,tokenizer,greedy,"aaaaaa",10);
        check(result.prompt_tokens==8 && result.generated_tokens.size()==1 && result.stop_reason=="context","exact context prompt");
        result=generate(repeat,tokenizer,greedy,"",0);
        check(result.generated_tokens.empty() && repeat.position()==0,"zero budget");
        rejects([&]{generate(repeat,tokenizer,greedy,"aaaaaaaa",3);},"long prompt rejects");
        check(repeat.position()==0,"long prompt state unchanged");
        std::cout<<"PASS framework "<<(gpu?"CUDA":"CPU")<<" ("<<assertions<<" assertions)\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
