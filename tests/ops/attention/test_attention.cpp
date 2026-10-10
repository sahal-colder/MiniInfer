#include "mini_infer/ops/attention/mini_infer_attention.h"
#include "mini_infer/ops/attention/mini_infer_attention_qkv.h"
#include "mini_infer/ops/embedding/mini_infer_embedding.h"
#include "mini_infer/ops/softmax/mini_infer_softmax.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <string>

using namespace mini_infer;
namespace {
auto cpu = std::make_shared<CPUDeviceAllocator>();
int checks = 0;
void require(bool value, const std::string& label) {
    if (!value) throw std::runtime_error(label);
    ++checks;
}
template<class F> void rejects(F f, const std::string& label) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, label);
}
void sync(cudaStream_t s, bool gpu) {
    if (gpu) check_cuda(cudaStreamSynchronize(s), "verification sync");
}
Tensor tensor(std::vector<int32_t> shape, int seed = 0) {
    Tensor t(DataType::FLOAT32, std::move(shape), cpu);
    for (std::size_t i = 0; i < t.size(); ++i)
        t.ptr<float>()[i] = float((int(i % 43)*7 + seed) % 31 - 15) / 19.0f;
    return t;
}
Tensor on(const Tensor& h, const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s) {
    return a->device_type() == DeviceType::CUDA ? h.to_cuda(a, s) : h.clone(a);
}
std::vector<double> values(const Tensor& t) {
    Tensor h = t.to_cpu();
    return {h.ptr<float>(), h.ptr<float>() + h.size()};
}
void close(const Tensor& t, const std::vector<double>& ref, const std::string& label,
           double tol = 4e-5) {
    const auto got = values(t);
    require(got.size() == ref.size(), label + " shape");
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (!std::isfinite(got[i]) || std::abs(got[i]-ref[i]) > tol*(1+std::abs(ref[i])))
            throw std::runtime_error(label + " idx=" + std::to_string(i) +
                " got=" + std::to_string(got[i]) + " ref=" + std::to_string(ref[i]));
    }
    ++checks;
}

// 独立 FP64 数学参考；不调用实现中的 Softmax、归约或索引函数。
std::vector<double> softmax_ref(const Tensor& input) {
    const auto data = values(input);
    const std::size_t width = input.dims().back();
    auto out = data;
    for (std::size_t start = 0; start < data.size(); start += width) {
        const double maximum = *std::max_element(data.begin()+start, data.begin()+start+width);
        double sum = 0;
        for (std::size_t i = 0; i < width; ++i) sum += std::exp(data[start+i]-maximum);
        for (std::size_t i = 0; i < width; ++i) out[start+i] = std::exp(data[start+i]-maximum)/sum;
    }
    return out;
}

void softmax_cases(const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s, bool large) {
    const bool gpu = a->device_type() == DeviceType::CUDA;
    const std::vector<int> widths = large ? std::vector<int>{3,65537}
        : std::vector<int>{1,3,31,127,128,129,513,8193};
    for (int width : widths) {
        const int rows = large && width == 3 ? 65539 : 6;
        Tensor h = tensor({2,rows/2,width}, 1);
        if (rows % 2) h = tensor({rows,width}, 1);
        // 大正数、大负数、相等值：检验稳定性与均匀概率。
        for (int i = 0; i < width; ++i) h.ptr<float>()[i] += 1000.0f;
        if (rows > 1) for (int i = 0; i < width; ++i) h.ptr<float>()[width+i] = -1000.0f;
        if (width > 1) h.ptr<float>()[2*width] = -std::numeric_limits<float>::infinity();
        const auto ref = softmax_ref(h);
        Tensor x = on(h,a,s), out(DataType::FLOAT32,h.dims(),a);
        SoftmaxLayer layer(a->device_type());
        layer.forward(x,out,s); sync(s,gpu);
        close(out,ref,"Softmax separate");
        // 掩码的 -inf 不用 close() 比较原始输入，单独确认零概率。
        if (width > 1) require(out.to_cpu().ptr<float>()[2*width] == 0,"masked probability zero");
        layer.forward(x,x,s); sync(s,gpu);
        close(x,ref,"Softmax inplace");
    }
    if (large) return;
    Tensor buf(DataType::FLOAT32,{10},a), shifted=buf.view(1,{9}), first=buf.view(0,{9});
    Tensor wrong(DataType::FLOAT32,{3,3},a), ints(DataType::INT32,{9},a), empty;
    SoftmaxLayer layer(a->device_type());
    rejects([&]{layer.forward(first,shifted,s);},"Softmax shifted alias rejected");
    rejects([&]{layer.forward(first,wrong,s);},"Softmax same count wrong rank rejected");
    rejects([&]{layer.forward(ints,first,s);},"Softmax dtype rejected");
    rejects([&]{layer.forward(empty,empty,s);},"Softmax empty rejected");
    if (gpu) {
        Tensor host=tensor({9});
        rejects([&]{layer.forward(host,first,s);},"Softmax mixed device rejected");
    } else rejects([&]{layer.forward(first,first,reinterpret_cast<cudaStream_t>(1));},"Softmax CPU stream rejected");
    rejects([]{(void)get_softmax_kernel(DeviceType::UNKNOWN);},"Softmax unknown backend rejected");
}

void embedding_cases(const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s, bool large) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    const int n=large ? 65539 : 5;
    for (int dim : (large ? std::vector<int>{5} : std::vector<int>{4,257})) {
        Tensor hw=tensor({8,dim},6), hi(DataType::INT32,{n},cpu);
        const int32_t ids[]={0,3,6,3,7};
        for (int i=0;i<n;++i) hi.ptr<int32_t>()[i]=ids[i%5];
        // 课件的三行查表实例。
        if (dim==4) {
            const float r0[]={1,2,3,4}, r3[]={4,3,1,7}, r6[]={6,1,3,4};
            std::copy_n(r0,4,hw.ptr<float>());
            std::copy_n(r3,4,hw.ptr<float>()+3*4);
            std::copy_n(r6,4,hw.ptr<float>()+6*4);
        }
        std::vector<double> ref;
        for (int i=0;i<n;++i) for (int j=0;j<dim;++j) ref.push_back(hw.ptr<float>()[hi.ptr<int32_t>()[i]*dim+j]);
        Tensor w=on(hw,a,s), id=on(hi,a,s), out(DataType::FLOAT32,{n,dim},a);
        EmbeddingLayer layer(w);
        layer.forward(id,out,s); sync(s,gpu);
        close(out,ref,"Embedding lookup and repeated ids",0);
        close(w,values(hw),"Embedding weight preserved",0);
        if (large) continue;
        for (int bad : {-1,8}) {
            hi.ptr<int32_t>()[2]=bad; id.copy_from(hi,s);
            Tensor sentinel=tensor({n,dim},2); out.copy_from(sentinel,s);
            // 非默认 stream 的 id 上传尚可能在进行；forward 必须同步验证。
            rejects([&]{layer.forward(id,out,s);},"Embedding bad id rejected");
            sync(s,gpu); close(out,values(sentinel),"Embedding rejected before writing",0);
        }
        hi.ptr<int32_t>()[2]=6; id.copy_from(hi,s);
        Tensor wrong(DataType::FLOAT32,{dim,n},a), wrong_ids(DataType::FLOAT32,{n},a), empty;
        rejects([&]{layer.forward(id,wrong,s);},"Embedding wrong output shape rejected");
        rejects([&]{layer.forward(wrong_ids,out,s);},"Embedding wrong id dtype rejected");
        rejects([&]{layer.forward(empty,out,s);},"Embedding empty ids rejected");
        Tensor overlap=w.view(0,{n,dim});
        rejects([&]{layer.forward(id,overlap,s);},"Embedding output/weight alias rejected");
        if (gpu) rejects([&]{layer.forward(hi,out,s);},"Embedding mixed device rejected");
        else rejects([&]{layer.forward(id,out,reinterpret_cast<cudaStream_t>(1));},"Embedding CPU stream rejected");
    }
    if (!large) {
        rejects([]{(void)get_embedding_kernel(DeviceType::UNKNOWN);},"Embedding unknown backend rejected");
        rejects([]{EmbeddingLayer layer(Tensor{});},"Embedding empty weight rejected");
    }
}

struct AttentionReference { std::vector<double> probabilities, output; };
AttentionReference attention_ref(const Tensor& q,const Tensor& k,const Tensor& v,
                                 int heads,int kv_heads,int length,int capacity) {
    const auto query=values(q),keys=values(k),vals=values(v);
    const int hd=q.dim(0)/heads;
    AttentionReference ref{std::vector<double>(std::size_t(heads)*capacity,1234.0),
                           std::vector<double>(q.size())};
    for (int h=0;h<heads;++h) {
        std::vector<double> logits(length);
        const int kvh=h/(heads/kv_heads);
        for (int t=0;t<length;++t) {
            double dot=0;
            for (int i=0;i<hd;++i) dot+=query[h*hd+i]*keys[std::size_t(t)*k.dim(1)+kvh*hd+i];
            logits[t]=dot/std::sqrt(double(hd));
        }
        const double m=*std::max_element(logits.begin(),logits.end());
        double den=0; for (double x : logits) den+=std::exp(x-m);
        for (int t=0;t<length;++t) {
            const double p=std::exp(logits[t]-m)/den;
            ref.probabilities[std::size_t(h)*capacity+t]=p;
            for (int i=0;i<hd;++i) ref.output[h*hd+i]+=p*vals[std::size_t(t)*v.dim(1)+kvh*hd+i];
        }
    }
    return ref;
}

void attention_cases(const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s,bool large) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    struct Case { int heads,kv_heads,hd,length; bool shifted; };
    const std::vector<Case> cases=large ? std::vector<Case>{{4,2,8,8193,false},{65539,1,1,3,false}}
        : std::vector<Case>{{2,2,1,1,false},{2,2,3,3,false},{4,2,4,129,false},
          {4,1,5,257,false},{3,3,7,9,false},{4,2,8,17,true},
          {2,2,128,7,false},{2,1,129,7,false},{4,2,7,1025,false}};
    for (const auto& c : cases) {
        const int dim=c.heads*c.hd,kv_dim=c.kv_heads*c.hd,cap=c.length+3;
        Tensor hq=tensor({dim},2),hk=tensor({c.length,kv_dim},4),hv=tensor({c.length,kv_dim},9);
        Tensor q=on(hq,a,s),k=on(hk,a,s),v=on(hv,a,s);
        // 明确强制 query/key 起点错位，覆盖 float4 的地址检查。
        Tensor qbase,kbase;
        if (c.shifted) {
            qbase=Tensor(DataType::FLOAT32,{dim+1},a); q=qbase.view(1,{dim}); q.copy_from(hq,s);
            kbase=Tensor(DataType::FLOAT32,{c.length*kv_dim+1},a);
            k=kbase.view(1,{c.length,kv_dim}); k.copy_from(hk,s);
        }
        Tensor hs(DataType::FLOAT32,{c.heads,cap},cpu);
        std::fill_n(hs.ptr<float>(),hs.size(),1234.0f);
        Tensor scores=on(hs,a,s),out(DataType::FLOAT32,{dim},a);
        const auto ref=attention_ref(hq,hk,hv,c.heads,c.kv_heads,c.length,cap);
        get_attention_kernel(a->device_type())(q,k,v,scores,out,c.heads,c.kv_heads,s);
        sync(s,gpu);
        close(out,ref.output,"Attention MHA/GQA/MQA FP64 output");
        close(scores,ref.probabilities,"Attention probabilities and untouched tail");
        close(q,values(hq),"Attention query preserved",0);
        close(k,values(hk),"Attention key history preserved",0);
        close(v,values(hv),"Attention value history preserved",0);
    }
    if (large) return;
    // 多层偏移、历史 query 的因果截断、缓存 metadata 不改变。
    KVCache cache(2,8,4,a);
    Tensor hq=tensor({8},1),hk=tensor({4,4},2),hv=tensor({4,4},3);
    Tensor q=on(hq,a,s),k=on(hk,a,s),v=on(hv,a,s);
    cache.append(1,k,v,s);
    Tensor hs(DataType::FLOAT32,{4,8},cpu); std::fill_n(hs.ptr<float>(),hs.size(),1234.0f);
    Tensor scores=on(hs,a,s),out(DataType::FLOAT32,{8},a);
    MultiHeadAttention attention(4,2,2);
    attention.forward(q,cache,1,1,scores,out,s); sync(s,gpu);
    const auto ref=attention_ref(hq,hk,hv,4,2,2,8);
    close(out,ref.output,"cache layer offset and causal prefix");
    close(scores,ref.probabilities,"causal prefix probability/tail");
    require(cache.valid_length(0)==0 && cache.valid_length(1)==4,"Attention preserves cache metadata");
    rejects([&]{attention.forward(q,cache,0,0,scores,out,s);},"empty layer rejected");
    rejects([&]{attention.forward(q,cache,1,4,scores,out,s);},"uncommitted position rejected");
    rejects([&]{attention.forward(q,cache,2,0,scores,out,s);},"invalid layer rejected");
    rejects([&]{attention.forward(q,cache,1,-1,scores,out,s);},"negative position rejected");
    Tensor tiny(DataType::FLOAT32,{4,1},a);
    rejects([&]{attention.forward(q,cache,1,1,tiny,out,s);},"score capacity too short rejected");
    rejects([&]{attention.forward(q,cache,1,1,scores,q,s);},"Attention query/output alias rejected");
    auto history=cache.history(1);
    Tensor cache_out=history.first.view(0,{8});
    rejects([&]{attention.forward(q,cache,1,1,scores,cache_out,s);},"Attention cache/output alias rejected");
    Tensor future_out=history.first.view(8,{8});
    rejects([&]{attention.forward(q,cache,1,1,scores,future_out,s);},"Attention future cache/output alias rejected");
    cache.append(0,k,v,s);
    auto other_history=cache.history(0);
    Tensor other_out=other_history.second.view(0,{8});
    rejects([&]{attention.forward(q,cache,1,1,scores,other_out,s);},"Attention other layer/output alias rejected");
    Tensor shared(DataType::FLOAT32,{32},a),shared_out=shared.view(0,{8});
    Tensor shared_scores=shared.view(0,{4,8});
    rejects([&]{attention.forward(q,cache,1,1,shared_scores,shared_out,s);},"score/output overlap rejected");
    Tensor wrong(DataType::FLOAT32,{2,4},a),ints(DataType::INT32,{8},a);
    rejects([&]{attention.forward(wrong,cache,1,1,scores,out,s);},"query rank rejected");
    rejects([&]{attention.forward(ints,cache,1,1,scores,out,s);},"query dtype rejected");
    rejects([&]{MultiHeadAttention wrong(4,4,2);wrong.forward(q,cache,1,1,scores,out,s);},"cache dimensions rejected");
    if (gpu) rejects([&]{attention.forward(hq,cache,1,1,scores,out,s);},"Attention mixed device rejected");
    else rejects([&]{attention.forward(q,cache,1,1,scores,out,reinterpret_cast<cudaStream_t>(1));},"Attention CPU stream rejected");
    rejects([]{MultiHeadAttention a(3,2,4);},"nondivisible head counts rejected");
    rejects([]{MultiHeadAttention a(0,1,4);},"zero head count rejected");
    rejects([]{MultiHeadAttention a(4,1,0);},"zero head size rejected");
    rejects([]{MultiHeadAttention a(4,1,std::numeric_limits<int32_t>::max());},"head dimension overflow rejected");
    rejects([]{(void)get_attention_kernel(DeviceType::UNKNOWN);},"Attention unknown backend rejected");
}

void pipeline_case(const std::shared_ptr<DeviceAllocator>& a,cudaStream_t s) {
    const bool gpu=a->device_type()==DeviceType::CUDA;
    Tensor hw=tensor({8,4},5),hi(DataType::INT32,{2},cpu);
    hi.ptr<int32_t>()[0]=0;hi.ptr<int32_t>()[1]=3;
    Tensor table=on(hw,a,s),ids=on(hi,a,s),embedded(DataType::FLOAT32,{2,4},a);
    EmbeddingLayer embedding(table);embedding.forward(ids,embedded,s);
    Tensor identity(DataType::FLOAT32,{4,4},cpu);std::fill_n(identity.ptr<float>(),16,0.0f);
    for(int i=0;i<4;++i) identity.ptr<float>()[i*4+i]=1;
    Tensor w=on(identity,a,s);AttentionQKV qkv({w,w,w});KVCache cache(1,8,4,a);
    Tensor q(DataType::FLOAT32,{4},a),scores(DataType::FLOAT32,{2,8},a),out(DataType::FLOAT32,{4},a);
    Tensor norm_weight(DataType::FLOAT32,{4},cpu);std::fill_n(norm_weight.ptr<float>(),4,1.0f);
    Tensor nw=on(norm_weight,a,s),normalized(DataType::FLOAT32,{4},a);
    RMSNormLayer norm(a->device_type());norm.set_weight(0,nw);
    for(int pos=0;pos<2;++pos) {
        norm.set_input(0,embedded.view(pos*4,{4}));norm.set_output(0,normalized);norm.forward(s);
        qkv.forward(normalized,q,cache,0,pos,s);cache.commit(0,pos);
    }
    // 原始投影链的组件测试；没有 RoPE，因此不宣称完整 Llama 前向。
    MultiHeadAttention attention(2,2,2);attention.forward(q,cache,0,1,scores,out,s);
    Tensor projected(DataType::FLOAT32,{4},a),residual(DataType::FLOAT32,{4},a);
    MatmulLayer wo(a->device_type(),4,4);wo.set_weight(0,w);wo.set_input(0,out);wo.set_output(0,projected);wo.forward(s);
    Tensor x=embedded.view(4,{4});AddLayer add(a->device_type());
    add.set_input(0,x);add.set_input(1,projected);add.set_output(0,residual);add.forward(s);sync(s,gpu);
    Tensor hk(DataType::FLOAT32,{2,4},cpu);
    for(int row=0;row<2;++row) {
        double square=0;for(int i=0;i<4;++i) square+=double(hw.ptr<float>()[hi.ptr<int32_t>()[row]*4+i])*hw.ptr<float>()[hi.ptr<int32_t>()[row]*4+i];
        for(int i=0;i<4;++i) hk.ptr<float>()[row*4+i]=float(hw.ptr<float>()[hi.ptr<int32_t>()[row]*4+i]/std::sqrt(square/4+1e-5));
    }
    Tensor hq=hk.view(4,{4});const auto ref=attention_ref(hq,hk,hk,2,2,2,8);
    auto result=ref.output;for(int i=0;i<4;++i) result[i]+=hw.ptr<float>()[3*4+i];
    close(residual,result,"Embedding -> RMSNorm -> QKV -> cache -> Attention -> Wo -> Add");
    // 复用的 AddLayer 仍保留旧接口的输出不重叠规则。
    add.set_output(0,x);rejects([&]{add.forward(s);},"existing AddLayer alias contract preserved");
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
    cudaStream_t s=nullptr;
    try {
        std::shared_ptr<DeviceAllocator> a=gpu ? std::static_pointer_cast<DeviceAllocator>(std::make_shared<CUDADeviceAllocator>()) : cpu;
        if(gpu && !default_stream) check_cuda(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"create nonblocking stream");
        softmax_cases(a,s,large);embedding_cases(a,s,large);attention_cases(a,s,large);
        if(!large) pipeline_case(a,s);
        sync(s,gpu);
        if(s) check_cuda(cudaStreamDestroy(s),"destroy stream");
        std::cout<<"PASS: lesson18 "<<(gpu?"CUDA":"CPU")<<(large?" large":"")
                 <<" ("<<checks<<" assertions)\n";
        return 0;
    } catch(const std::exception& e) {
        if(gpu) cudaStreamSynchronize(s);
        if(s) cudaStreamDestroy(s);
        std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;
    }
}
