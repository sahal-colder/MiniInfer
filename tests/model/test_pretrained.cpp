#include "mini_infer/generation/mini_infer_generation.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

extern "C" {
void* ref_model_open(const char*);
const float* ref_forward(void*,int,int);
void ref_model_close(void*);
void* ref_tokenizer_open(const char*,int);
int ref_encode(void*,const char*,int*);
void ref_tokenizer_close(void*);
}
int main(int argc,char** argv) {
    using namespace mini_infer;
    try {
        if (argc < 3) throw std::invalid_argument("expected model and tokenizer paths");
        const std::string model_path=argv[1],tokenizer_path=argv[2];
        const bool gpu=argc>3;
        auto& out=std::cout;
        std::shared_ptr<DeviceAllocator> allocator=std::make_shared<CPUDeviceAllocator>();
        if(gpu) allocator=std::make_shared<CUDADeviceAllocator>();
        LlamaModel model(model_path,allocator);
        Tokenizer tokenizer(tokenizer_path,model.config().vocab_size);
        void* reference=ref_model_open(model_path.c_str());
        void* reference_tokenizer=ref_tokenizer_open(tokenizer_path.c_str(),model.config().vocab_size);
        const auto vocab=model.config().vocab_size;
        const std::vector<std::string> prompts={"","Once upon a time","Hello world!","你好，世界🙂","  spaces\nand\ttabs","<s> literal"};
        for(const auto& text:prompts) {
            auto actual=tokenizer.encode(text);
            std::vector<int> expected(text.size()+4);
            expected.resize(ref_encode(reference_tokenizer,text.c_str(),expected.data()));
            if(actual!=expected) throw std::runtime_error("reference tokenizer mismatch: "+text);
        }
        const auto tokens=tokenizer.encode("Once upon a time");
        Sampler sampler;
        double max_error=0;
        int token=tokens[0];
        int previous=token;
        out<<"Pretrained "<<(gpu?"CUDA":"CPU")<<": ";
        for(int pos=0;pos<80;++pos) {
            const auto& actual=model.forward(token);
            const float* expected=ref_forward(reference,token,pos);
            for(int i=0;i<vocab;++i) {
                const double error=std::abs(double(actual.ptr<float>()[i])-expected[i]);
                max_error=std::max(max_error,error);
                if(!std::isfinite(error) || error>5e-4*(1+std::abs(expected[i])))
                    throw std::runtime_error("reference logits mismatch at pos="+std::to_string(pos));
            }
            int next=sampler.sample(actual);
            const int expected_next=static_cast<int>(std::max_element(expected,expected+vocab)-expected);
            if(next!=expected_next) throw std::runtime_error("reference greedy token mismatch");
            if(pos+1<static_cast<int>(tokens.size())) next=tokens[pos+1];
            out<<tokenizer.decode(previous,next);previous=next;token=next;
        }
        out<<"\nPASS pretrained: 80 positions x "<<vocab
           <<" logits, greedy tokens identical, max_abs_error="<<max_error<<"\n";
        ref_tokenizer_close(reference_tokenizer);ref_model_close(reference);
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
