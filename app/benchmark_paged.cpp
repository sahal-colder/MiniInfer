#include "mini_infer/model/mini_infer_model.h"
#include <cuda_runtime_api.h>
#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>
#include <vector>
namespace {
using Clock=std::chrono::steady_clock;
using mini_infer::PagedModelStats;
double elapsed(Clock::time_point start) {return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
std::size_t number(const std::string& text) {
    std::size_t value=0;const auto parsed=std::from_chars(text.data(),text.data()+text.size(),value);
    if(parsed.ec!=std::errc{}||parsed.ptr!=text.data()+text.size()||!value) throw std::invalid_argument("Expected positive integer: "+text);
    return value;
}
std::string quoted(const std::string& text) {
    std::ostringstream out;out<<'"';
    for(unsigned char c:text) {
        if(c=='"'||c=='\\') out<<'\\'<<char(c);
        else if(c<32) out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<char(c);
    }
    out<<'"';return out.str();
}
int32_t argmax(const mini_infer::Tensor& logits) {
    const auto* values=logits.ptr<float>();
    return static_cast<int32_t>(std::max_element(values,values+logits.size())-values);
}
void stats(std::ostream& out,const PagedModelStats& s) {
    out<<'{';bool comma=false;
    const auto field=[&](const char* name,auto value){if(comma)out<<',';comma=true;out<<quoted(name)<<':'<<value;};
#define FIELD(name) field(#name,s.name)
    FIELD(computed_tokens);FIELD(prefix_hit_tokens);FIELD(last_prefix_hit_tokens);FIELD(demand_fault_pages);
    FIELD(prefetched_pages);FIELD(prefetch_windows);FIELD(prefetch_skipped);FIELD(gpu_evicted_pages);
    FIELD(ssd_spilled_pages);FIELD(prefix_evictions);FIELD(layer_calls);FIELD(prefetch_wait_ns);
    FIELD(page_bytes);FIELD(tokens_per_page);FIELD(pages);FIELD(gpu_limit_bytes);FIELD(host_limit_bytes);FIELD(ssd_limit_bytes);
    FIELD(gpu_reserved_bytes);FIELD(host_reserved_bytes);FIELD(ssd_reserved_bytes);
    FIELD(gpu_peak_bytes);FIELD(host_peak_bytes);FIELD(ssd_peak_bytes);FIELD(staging_bytes);
    FIELD(ssd_read_bytes);FIELD(ssd_written_bytes);FIELD(ssd_record_writes);FIELD(uring_submissions);
#undef FIELD
    out<<'}';
}
struct Request {
    double ttft=0,total=0;
    std::vector<double> decode;
    std::vector<int32_t> tokens;
    std::vector<float> first_logits;
    PagedModelStats before,prefill,usage;
};
Request request(mini_infer::LlamaModel& model,const std::vector<int32_t>& prompt,std::size_t count) {
    Request result;result.before=model.paged_stats();const auto all=Clock::now();
    const auto& first=model.prefill(prompt);result.tokens.push_back(argmax(first));result.ttft=elapsed(all);result.prefill=model.paged_stats();
    result.first_logits.assign(first.ptr<float>(),first.ptr<float>()+first.size());
    for(std::size_t i=1;i<count;++i) {
        const auto start=Clock::now();result.tokens.push_back(argmax(model.forward(result.tokens.back())));
        result.decode.push_back(elapsed(start));
    }
    result.total=result.ttft;for(auto value:result.decode)result.total+=value;
    result.usage=model.paged_stats();return result;
}
void output(std::ostream& out,const Request& r,std::size_t prompt_count) {
    double decode_time=0;for(auto x:r.decode)decode_time+=x;
    out<<"{\"ttft_ms\":"<<r.ttft<<",\"request_ms\":"<<r.total
       <<",\"decode_tokens_s\":"<<(decode_time>0?1000*r.decode.size()/decode_time:0)
       <<",\"end_to_end_output_tokens_s\":"<<1000*r.tokens.size()/r.total
       <<",\"logical_input_tokens\":"<<prompt_count<<",\"tokens\":[";
    for(std::size_t i=0;i<r.tokens.size();++i){if(i)out<<',';out<<r.tokens[i];}
    out<<"],\"decode_samples_ms\":[";
    for(std::size_t i=0;i<r.decode.size();++i){if(i)out<<',';out<<r.decode[i];}
    std::uint64_t hash=1469598103934665603ULL;
    const auto* bytes=reinterpret_cast<const unsigned char*>(r.first_logits.data());
    for(std::size_t i=0;i<r.first_logits.size()*sizeof(float);++i){hash^=bytes[i];hash*=1099511628211ULL;}
    out<<"],\"first_logit_hash\":"<<quoted(std::to_string(hash))<<",\"stats_before\":";stats(out,r.before);
    out<<",\"stats_after_prefill\":";stats(out,r.prefill);out<<",\"stats\":";stats(out,r.usage);out<<'}';
}
}
int main(int argc,char** argv) {try {
    std::string path,variant="paged-tiered";std::size_t prompt_count=48,generated=12,page_tokens=8,gpu=1<<20,host=512<<10;
    bool gpu_override=false;
    for(int i=1;i<argc;i+=2) {
        if(i+1==argc)throw std::invalid_argument("Missing benchmark argument value");
        const std::string key=argv[i],value=argv[i+1];
        if(key=="--model")path=value;
        else if(key=="--variant")variant=value;
        else if(key=="--prompt")prompt_count=number(value);
        else if(key=="--generate")generated=number(value);
        else if(key=="--page-tokens")page_tokens=number(value);
        else if(key=="--gpu-bytes") {gpu=number(value);gpu_override=true;}
        else if(key=="--host-bytes")host=number(value);
        else throw std::invalid_argument("Unknown benchmark argument: "+key);
    }
    if(path.empty())throw std::invalid_argument("--model is required");
    if(variant!="legacy"&&variant!="paged-resident"&&variant!="paged-tiered"&&variant!="paged-demand"&&variant!="paged-async")throw std::invalid_argument("Unknown variant");
    const auto context=Clock::now();
    mini_infer::check_cuda(cudaSetDevice(0),"Benchmark device");
    mini_infer::check_cuda(cudaFree(nullptr),"Benchmark context initialization");
    mini_infer::check_cuda(cudaDeviceSynchronize(),"Benchmark context ready");const auto context_ms=elapsed(context);
    auto allocator=std::make_shared<mini_infer::CUDADeviceAllocator>();mini_infer::ModelOptions options;
    options.kv_mode=variant=="legacy"?mini_infer::ModelKVMode::CONTIGUOUS:mini_infer::ModelKVMode::PAGED;
    options.paged.tokens_per_page=page_tokens;
    if(variant=="paged-resident")options.paged.resident_layers=0;
    else {
        options.paged.enable_async_kv_pipeline=variant=="paged-async";
        options.paged.gpu_bytes=options.paged.enable_async_kv_pipeline && !gpu_override ? 0 : gpu;
        options.paged.host_bytes=host;
    }
    options.paged.prefetch=variant!="paged-demand";
    rusage before{},after{};getrusage(RUSAGE_SELF,&before);
    const auto initialized=Clock::now();mini_infer::LlamaModel model(path,allocator,options);
    mini_infer::check_cuda(cudaDeviceSynchronize(),"Benchmark model ready");const auto init_ms=elapsed(initialized);
    getrusage(RUSAGE_SELF,&after);
    if(prompt_count>std::size_t(model.config().seq_len)||generated>std::size_t(model.config().seq_len)-prompt_count)
        throw std::invalid_argument("Benchmark exceeds model context");
    std::vector<int32_t> prompt;prompt.reserve(prompt_count);
    for(std::size_t i=0;i<prompt_count;++i)prompt.push_back(static_cast<int32_t>((i*17+3)%model.config().vocab_size));
    const auto miss=request(model,prompt,generated);model.reset();const auto hit=request(model,prompt,generated);
    const bool same=miss.first_logits.size()==hit.first_logits.size()&&
        !std::memcmp(miss.first_logits.data(),hit.first_logits.data(),miss.first_logits.size()*sizeof(float));
    const auto& loader=model.weight_io_stats();
    std::cout<<std::setprecision(12)<<"{\"variant\":"<<quoted(variant)<<",\"context_init_ms\":"<<context_ms
        <<",\"model_init_ms\":"<<init_ms<<",\"startup_through_first_token_ms\":"<<context_ms+init_ms+miss.ttft
        <<",\"initialization_major_faults\":"<<after.ru_majflt-before.ru_majflt
        <<",\"weight_backend\":"<<quoted(loader.backend)<<",\"weight_fallback\":"<<quoted(loader.fallback_reason)
        <<",\"weight_tensor_bytes\":"<<(loader.backend=="legacy-mmap"?std::string("null"):std::to_string(loader.tensor_bytes))
        <<",\"weight_staging_bytes\":"<<(loader.backend=="legacy-mmap"?std::string("null"):std::to_string(loader.staging_bytes))
        <<",\"paged\":"<<(model.uses_paged_kv()?"true":"false")
        <<",\"prefix_logits_bitwise_equal\":"<<(same?"true":"false")<<",\"miss\":";
    output(std::cout,miss,prompt_count);std::cout<<",\"hit\":";output(std::cout,hit,prompt_count);std::cout<<"}\n";
    return same&&miss.tokens==hit.tokens?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
