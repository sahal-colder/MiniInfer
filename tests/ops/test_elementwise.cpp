#include "mini_infer/ops/add/mini_infer_add.h"
#include "mini_infer/ops/embedding/mini_infer_embedding.h"
#include "mini_infer/ops/softmax/mini_infer_softmax.h"
#include "mini_infer/ops/base/mini_infer_layer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>

using namespace mini_infer;
namespace {
auto cpu=std::make_shared<CPUDeviceAllocator>();
int checks=0;
void require(bool condition,const std::string& label) {
    if(!condition) throw std::runtime_error(label);
    ++checks;
}
template<class F> void rejects(F f,const std::string& label) {
    bool caught=false;try {f();} catch(const std::exception&) {caught=true;}
    require(caught,label);
}
void sync(cudaStream_t s,bool gpu) {
    if(gpu) check_cuda(cudaStreamSynchronize(s),"test sync");
}
Tensor tensor(std::vector<int32_t> shape,int seed=0) {
    Tensor t(DataType::FLOAT32,std::move(shape),cpu);
    for(std::size_t i=0;i<t.size();++i) t.ptr<float>()[i]=float((int(i%47)*11+seed)%37-18)/13.0f;
    return t;
}
Tensor on(const Tensor& host,const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s) {
    return a->device_type()==DeviceType::CUDA ? host.to_cuda(a,s) : host.clone(a);
}
std::vector<double> values(const Tensor& t) {
    Tensor host=t.to_cpu();return {host.ptr<float>(),host.ptr<float>()+host.size()};
}
void close(const Tensor& t,const std::vector<double>& expected,const std::string& label,double tol=1e-6) {
    const auto got=values(t);require(got.size()==expected.size(),label+" shape");
    for(std::size_t i=0;i<got.size();++i) {
        if(!std::isfinite(got[i]) || std::abs(got[i]-expected[i])>tol*(1+std::abs(expected[i])))
            throw std::runtime_error(label+" idx="+std::to_string(i));
    }
    ++checks;
}

void add_cases(const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s,bool large) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    const std::vector<std::vector<int32_t>> shapes=large
        ? std::vector<std::vector<int32_t>>{{65535*32+35}}
        : std::vector<std::vector<int32_t>>{{1},{31},{32},{33},{1023},{1024},{1025},{2,3,7}};
    ElementwiseAdd add(a->device_type());
    for(const auto& shape:shapes) {
        Tensor ha=tensor(shape,3),hb=tensor(shape,7);
        std::vector<double> expected(ha.size());
        for(std::size_t i=0;i<expected.size();++i)
            expected[i]=double(ha.ptr<float>()[i])+hb.ptr<float>()[i];
        Tensor x=on(ha,a,s),y=on(hb,a,s),out(DataType::FLOAT32,shape,a);
        add.forward(x,y,out,s);sync(s,gpu);
        close(out,expected,"Add FP64 result and thread tail");
        close(x,values(ha),"Add first input preserved",0);
        close(y,values(hb),"Add second input preserved",0);
        // 两个输入共享只读存储合法。
        for(std::size_t i=0;i<expected.size();++i) expected[i]=2.0*ha.ptr<float>()[i];
        get_elementwise_add_kernel(a->device_type())(x,x,out,s);sync(s,gpu);
        close(out,expected,"Add shared read-only inputs");
        // 原 AddLayer 的绑定方式及数值结果继续可用。
        Tensor legacy_out(DataType::FLOAT32,shape,a);AddLayer legacy(a->device_type());
        legacy.set_input(0,x);legacy.set_input(1,x);legacy.set_output(0,legacy_out);legacy.forward(s);sync(s,gpu);
        close(legacy_out,expected,"legacy AddLayer unchanged");
    }
    if(large) return;
    Tensor host=tensor({9},5),x=on(host,a,s),y=on(host,a,s),out=on(host,a,s);
    const auto original=values(host);
    rejects([&]{add.forward(x,y,x,s);},"exact output/input alias rejected");
    rejects([&]{add.forward(x,y,y,s);},"second input alias rejected");
    Tensor all(DataType::FLOAT32,{10},a),first=all.view(0,{9}),shifted=all.view(1,{9});
    rejects([&]{add.forward(first,y,shifted,s);},"partial output/input alias rejected");
    Tensor wrong_shape(DataType::FLOAT32,{3,3},a),ints(DataType::INT32,{9},a),empty;
    rejects([&]{add.forward(x,wrong_shape,out,s);},"Add shape mismatch rejected");
    rejects([&]{add.forward(x,y,wrong_shape,s);},"Add output shape rejected");
    rejects([&]{add.forward(ints,y,out,s);},"Add integer input rejected");
    rejects([&]{add.forward(x,y,ints,s);},"Add integer output rejected");
    rejects([&]{add.forward(empty,y,out,s);},"Add empty rejected");
    if(gpu) rejects([&]{add.forward(host,y,out,s);},"Add mixed device rejected");
    else rejects([&]{add.forward(x,y,out,reinterpret_cast<cudaStream_t>(1));},"CPU Add stream rejected");
    sync(s,gpu);close(out,original,"invalid Add parameters do not write output",0);
    rejects([]{(void)get_elementwise_add_kernel(DeviceType::UNKNOWN);},"unknown Add selector rejected");
    rejects([]{ElementwiseAdd bad(DeviceType::UNKNOWN);},"unknown Add constructor rejected");
}

void course_lookup(const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    Tensor table(DataType::FLOAT32,{8,4},cpu),ids(DataType::INT32,{3},cpu);
    std::fill_n(table.ptr<float>(),table.size(),0.0f);
    const float rows[][4]={{1,2,3,4},{4,3,1,7},{6,1,3,4}};
    const int32_t id_values[]={0,3,6};
    std::vector<double> expected;
    for(int t=0;t<3;++t) {
        ids.ptr<int32_t>()[t]=id_values[t];
        std::copy_n(rows[t],4,table.ptr<float>()+id_values[t]*4);
        expected.insert(expected.end(),rows[t],rows[t]+4);
    }
    Tensor weight=on(table,a,s),input=on(ids,a,s),output(DataType::FLOAT32,{3,4},a);
    EmbeddingLayer(weight).forward(input,output,s);sync(s,gpu);
    close(output,expected,"lesson19 exact ids [0,3,6] lookup",0);
}

void course_softmax(const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    // 两头 scores + 因果 mask：Add 和 Softmax 使用同一 stream 排队。
    Tensor logits(DataType::FLOAT32,{2,4},cpu),mask(DataType::FLOAT32,{2,4},cpu);
    const float neg_inf=-std::numeric_limits<float>::infinity();
    const float data[]={-1000,-1001,-1002,-999,1000,1001,1002,999};
    const float mask_data[]={0,0,0,neg_inf,0,0,0,neg_inf};
    std::copy_n(data,8,logits.ptr<float>());std::copy_n(mask_data,8,mask.ptr<float>());
    Tensor x=on(logits,a,s),m=on(mask,a,s),scores(DataType::FLOAT32,{2,4},a);
    ElementwiseAdd(a->device_type()).forward(x,m,scores,s);
    SoftmaxLayer(a->device_type()).forward(scores,scores,s);sync(s,gpu);
    const double den=1+std::exp(-1.0)+std::exp(-2.0);
    const std::vector<double> expected={1/den,std::exp(-1.0)/den,std::exp(-2.0)/den,0,
                                       std::exp(-2.0)/den,std::exp(-1.0)/den,1/den,0};
    close(scores,expected,"per-head stable inplace Softmax after Add mask");
    const auto prob=values(scores);
    require(std::abs(prob[0]+prob[1]+prob[2]-1)<1e-6,"first head probability sum");
    require(std::abs(prob[4]+prob[5]+prob[6]-1)<1e-6,"second head probability sum");
    // cache_len+1 小于块大小且所有输入为很小的负数，仍须得到正确 max。
    for(int width:{1,3,31,32,127,128,129}) {
        Tensor h(DataType::FLOAT32,{width},cpu);
        for(int i=0;i<width;++i) h.ptr<float>()[i]=i==0 ? -1000.0f : -1001.0f;
        Tensor input=on(h,a,s),output(DataType::FLOAT32,{width},a);
        SoftmaxLayer(a->device_type()).forward(input,output,s);sync(s,gpu);
        const double sum=1+(width-1)*std::exp(-1.0);
        std::vector<double> ref(width,std::exp(-1.0)/sum);ref[0]=1/sum;
        close(output,ref,"negative logits and partial Softmax block");
    }
}
}  // namespace

int main(int argc,char** argv) {
    bool gpu=false,large=false,default_stream=false;
    for(int i=1;i<argc;++i) {
        const std::string arg=argv[i];
        if(arg=="--cuda") gpu=true;
        else if(arg=="--large") large=true;
        else if(arg=="--default-stream") default_stream=true;
        else {std::cerr<<"Unknown argument: "<<arg<<'\n';return 2;}
    }
    cudaStream_t stream=nullptr;
    try {
        std::shared_ptr<DeviceAllocator> a=gpu ? std::static_pointer_cast<DeviceAllocator>(std::make_shared<CUDADeviceAllocator>()) : cpu;
        if(gpu && !default_stream) check_cuda(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
        add_cases(a,stream,large);
        if(!large) {course_lookup(a,stream);course_softmax(a,stream);}
        sync(stream,gpu);if(stream) check_cuda(cudaStreamDestroy(stream),"destroy stream");
        std::cout<<"PASS: lesson19 "<<(gpu?"CUDA":"CPU")<<(large?" large":"")<<" ("<<checks<<" assertions)\n";
        return 0;
    } catch(const std::exception& e) {
        if(gpu) cudaStreamSynchronize(stream);
        if(stream) cudaStreamDestroy(stream);
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
