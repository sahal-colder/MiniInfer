#include "mini_infer/weights/mini_infer_mlp_weight_loader.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>

using namespace mini_infer;
namespace {
auto cpu = std::make_shared<CPUDeviceAllocator>();
int checks = 0;
void require(bool yes, const std::string& label) {
    if (!yes) throw std::runtime_error(label);
    ++checks;
}
template<class F> void rejects(F f, const std::string& label) {
    bool caught = false;
    try { f(); } catch (const std::exception&) { caught = true; }
    require(caught, label);
}
void sync(cudaStream_t s, bool gpu) {
    if (gpu) check_cuda(cudaStreamSynchronize(s), "test stream sync");
}
Tensor tensor(std::vector<int32_t> shape, int seed = 0) {
    Tensor t(DataType::FLOAT32, std::move(shape), cpu);
    for (std::size_t i = 0; i < t.size(); ++i)
        t.ptr<float>()[i] = float((int(i % 41) * 7 + seed) % 29 - 14) / 17.0f;
    return t;
}
Tensor on(const Tensor& h, const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s) {
    return a->device_type() == DeviceType::CUDA ? h.to_cuda(a, s) : h.clone(a);
}
void close(const Tensor& actual, const std::vector<double>& expected,
           const std::string& label, double tol = 3e-5) {
    Tensor h = actual.to_cpu();
    require(h.size() == expected.size(), label + " size");
    for (std::size_t i = 0; i < h.size(); ++i) {
        const double v = h.ptr<float>()[i], ref = expected[i];
        if (!std::isfinite(v) || std::abs(v - ref) > tol * (1.0 + std::abs(ref)))
            throw std::runtime_error(label + " idx=" + std::to_string(i) +
                " got=" + std::to_string(v) + " ref=" + std::to_string(ref));
    }
    ++checks;
}
std::vector<double> values(const Tensor& t) {
    Tensor h = t.to_cpu();
    return {h.ptr<float>(), h.ptr<float>() + h.size()};
}
std::vector<double> swiglu_ref(const Tensor& g, const Tensor& u) {
    std::vector<double> out(g.size());
    for (std::size_t i = 0; i < out.size(); ++i) {
        const double x = g.ptr<float>()[i];
        out[i] = (x / (1.0 + std::exp(-x))) * double(u.ptr<float>()[i]);
    }
    return out;
}
void swiglu_cases(const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s, bool large) {
    const bool gpu = a->device_type() == DeviceType::CUDA;
    const std::vector<int> sizes = large ? std::vector<int>{65535 * 128 + 3}
                                       : std::vector<int>{1, 2, 127, 128, 129, 32 * 151, 8193};
    for (int n : sizes) {
        Tensor hg = tensor({n}, 3), hu = tensor({n}, 9);
        const float extremes[] = {-100, -20, -1, 0, 1, 20, 100};
        for (int i = 0; i < std::min(n, 7); ++i) hg.ptr<float>()[i] = extremes[i];
        const auto ref = swiglu_ref(hg, hu);
        std::vector<SwiGLUKernel> kernels{get_swiglu_kernel(a->device_type())};
        if (gpu) kernels.push_back(swiglu_kernel_cuda_shared);
        else kernels.push_back(swiglu_kernel_cpu_reference);
        for (auto kernel : kernels) {
            Tensor g = on(hg, a, s), u = on(hu, a, s);
            Tensor out(DataType::FLOAT32, {n}, a);
            kernel(g, u, out, s);
            sync(s, gpu);
            close(out, ref, "SwiGLU separate", 3e-6);
            close(g, values(hg), "gate unchanged", 0);
            close(u, values(hu), "up unchanged", 0);
            kernel(g, u, g, s);
            sync(s, gpu);
            close(g, ref, "SwiGLU gate inplace", 3e-6);
            g.copy_from(hg, s);
            kernel(g, u, u, s);
            sync(s, gpu);
            close(u, ref, "SwiGLU up inplace", 3e-6);
            g.copy_from(hg, s);
            kernel(g, g, g, s);
            sync(s, gpu);
            close(g, swiglu_ref(hg, hg), "SwiGLU all same", 3e-6);
        }
    }
    if (large) return;
    Tensor all(DataType::FLOAT32, {9}, a);
    Tensor left = all.view(0, {8}), right = all.view(1, {8});
    Tensor h = tensor({8}), u = on(h, a, s), out(DataType::FLOAT32, {8}, a);
    sync(s, gpu);
    SwiGLULayer layer(a->device_type());
    rejects([&] { layer.forward(left, u, right, s); }, "shifted output rejected");
    Tensor wrong(DataType::FLOAT32, {2, 4}, a);
    rejects([&] { layer.forward(u, wrong, out, s); }, "same count different shape rejected");
    Tensor ints(DataType::INT32, {8}, a), empty;
    rejects([&] { layer.forward(u, ints, out, s); }, "int dtype rejected");
    rejects([&] { layer.forward(empty, u, out, s); }, "empty rejected");
    if (gpu) rejects([&] { layer.forward(h, u, out, s); }, "mixed device rejected");
    else rejects([&] { layer.forward(u, u, out, reinterpret_cast<cudaStream_t>(1)); },
                 "CPU stream rejected");
    rejects([&] { (void)get_swiglu_kernel(DeviceType::UNKNOWN); }, "unknown backend rejected");
    layer.forward(u, u, out, s);
    sync(s, gpu);
    close(out, swiglu_ref(h, h), "layer dispatch");
}
MLPWeights host_weights(int d, int h) {
    return {tensor({d}, 5), tensor({h, d}, 7), tensor({d, h}, 11), tensor({h, d}, 19)};
}
MLPWeights move_weights(const MLPWeights& w, const std::shared_ptr<DeviceAllocator>& a,
                        cudaStream_t s) {
    return {on(w.norm, a, s), on(w.w1, a, s), on(w.w2, a, s), on(w.w3, a, s)};
}
struct Reference { std::vector<double> norm, act, up, down, output; };
Reference mlp_ref(const Tensor& x, const MLPWeights& w, double eps) {
    const std::size_t d = w.norm.size(), h = std::size_t(w.w1.dim(0)), rows = x.size() / d;
    Reference r;
    r.norm.resize(rows*d); r.act.resize(rows*h); r.up.resize(rows*h);
    r.down.resize(rows*d); r.output.resize(rows*d);
    for (std::size_t row = 0; row < rows; ++row) {
        double square = 0;
        for (std::size_t j = 0; j < d; ++j) square += std::pow(double(x.ptr<float>()[row*d+j]), 2);
        const double scale = 1.0 / std::sqrt(square / d + eps);
        for (std::size_t j = 0; j < d; ++j)
            r.norm[row*d+j] = x.ptr<float>()[row*d+j] * scale * w.norm.ptr<float>()[j];
        for (std::size_t k = 0; k < h; ++k) {
            double g = 0, up = 0;
            for (std::size_t j = 0; j < d; ++j) {
                g += w.w1.ptr<float>()[k*d+j] * r.norm[row*d+j];
                up += w.w3.ptr<float>()[k*d+j] * r.norm[row*d+j];
            }
            r.up[row*h+k] = up;
            r.act[row*h+k] = (g / (1 + std::exp(-g))) * up;
        }
        for (std::size_t j = 0; j < d; ++j) {
            double down = 0;
            for (std::size_t k = 0; k < h; ++k) down += w.w2.ptr<float>()[j*h+k] * r.act[row*h+k];
            r.down[row*d+j] = down;
            r.output[row*d+j] = x.ptr<float>()[row*d+j] + down;
        }
    }
    return r;
}
void mlp_cases(const std::shared_ptr<DeviceAllocator>& a, cudaStream_t s) {
    const bool gpu = a->device_type() == DeviceType::CUDA;
    constexpr int d = 5, h = 7;
    constexpr float eps = 1e-5f;
    for (const auto& shape : std::vector<std::vector<int32_t>>{{d}, {2,d}, {2,3,d}}) {
        Tensor hx = tensor(shape, 2);
        MLPWeights hw = host_weights(d,h), w = move_weights(hw,a,s);
        Tensor x = on(hx,a,s), y(DataType::FLOAT32, shape, a);
        MLP mlp(w,eps); MLPWorkspace work(shape,h,a);
        const auto ref = mlp_ref(hx,hw,eps);
        const float* workspace_address = work.w1_output.ptr<float>();
        mlp.forward(x,y,work,s);
        sync(s,gpu);
        close(y,ref.output,"MLP output");
        close(work.norm_output,ref.norm,"MLP norm");
        close(work.w1_output,ref.act,"MLP gate activation");
        close(work.w3_output,ref.up,"MLP up");
        close(work.w2_output,ref.down,"MLP down");
        close(x,values(hx),"MLP separate preserves input",0);
        mlp.forward(x,x,work,s);
        sync(s,gpu);
        close(x,ref.output,"MLP residual inplace");
        Tensor second_input = x.to_cpu();
        const auto second = mlp_ref(second_input,hw,eps);
        mlp.forward(x,x,work,s);
        sync(s,gpu);
        close(x,second.output,"MLP work reuse");
        require(workspace_address == work.w1_output.ptr<float>(),"workspace reused");
        close(w.norm,values(hw.norm),"norm weight unchanged",0);
        close(w.w1,values(hw.w1),"W1 unchanged",0);
        close(w.w2,values(hw.w2),"W2 unchanged",0);
        close(w.w3,values(hw.w3),"W3 unchanged",0);
    }
    MLPWeights hw = host_weights(d,h), w = move_weights(hw,a,s);
    Tensor hx = tensor({d}), x = on(hx,a,s), y = on(hx,a,s);
    MLP mlp(w); MLPWorkspace work({d},h,a);
    sync(s,gpu);
    const auto before = values(y);
    const auto reject_unchanged = [&](auto f, const std::string& label) {
        rejects(f,label); sync(s,gpu); close(y,before,label+" no writes",0);
    };
    auto broken = work;
    broken.w1_output = broken.w3_output;
    reject_unchanged([&] { mlp.forward(x,y,broken,s); },"workspace self overlap");
    broken = work; broken.norm_output = x;
    reject_unchanged([&] { mlp.forward(x,y,broken,s); },"workspace input overlap");
    broken = work; broken.w3_output = w.w2.view(0,{h});
    reject_unchanged([&] { mlp.forward(x,y,broken,s); },"workspace cross weight overlap");
    broken = work; broken.w2_output = tensor({d+1});
    reject_unchanged([&] { mlp.forward(x,y,broken,s); },"late down workspace shape error");
    rejects([&] { mlp.forward(x,w.norm,work,s); },"output overlaps norm weight");
    Tensor base(DataType::FLOAT32,{d+1},a);
    Tensor shifted_input=base.view(0,{d}), shifted_output=base.view(1,{d});
    rejects([&] { mlp.forward(shifted_input,shifted_output,work,s); },"partial residual overlap");
    MLPWeights bad=w; bad.w2=w.w1;
    rejects([&] { MLP invalid(bad); },"W2 transpose rejected");
    rejects([&] { MLP invalid(w,0); },"zero epsilon rejected");
    rejects([&] { MLP invalid(w,std::numeric_limits<float>::quiet_NaN()); },"NaN epsilon rejected");
    rejects([&] { MLPWorkspace invalid({},h,a); },"workspace missing shape");
    rejects([&] { MLPWorkspace invalid({d},h,nullptr); },"workspace missing allocator");
    rejects([&] { MLPWorkspace invalid({d},0,a); },"workspace invalid hidden");
    if (!gpu) rejects([&] { mlp.forward(x,y,work,reinterpret_cast<cudaStream_t>(1)); },"CPU MLP stream");
    // 构造函数共享并保活权重；局部权重对象离开作用域后仍可计算。
    MLP owned = [&] { auto local=move_weights(hw,a,s); return MLP(local); }();
    owned.forward(x,y,work,s); sync(s,gpu);
    close(y,mlp_ref(hx,hw,eps).output,"weights outlive caller handles");
}
void loader_cases(const std::shared_ptr<DeviceAllocator>& a) {
    const ModelConfig c{4,7,2,2,1,-3,8};
    const auto layout=make_mlp_weight_layout(c);
    require(layout.norm_begin==116 && layout.w1_begin==124 && layout.w2_begin==180 &&
            layout.w3_begin==236 && layout.end==292,"manual GQA layout offsets");
    auto mha=c; mha.kv_head_num=2;
    require(make_mlp_weight_layout(mha).end==324,"MHA layout");
    auto pos=c; pos.vocab_size=3;
    require(make_mlp_weight_layout(pos).end==292,"vocab sign preserves layout");
    auto bad=c; bad.head_num=3;
    rejects([&] { make_mlp_weight_layout(bad); },"bad GQA config");
    auto big=c; big.dim=std::numeric_limits<int32_t>::max(); big.head_num=1;
    big.hidden_dim=big.dim; big.layer_num=big.dim;
    rejects([&] { make_mlp_weight_layout(big); },"layout overflow rejected");
    rejects([&] { MLPWeightLoader invalid(nullptr); },"null mapping rejected");
    std::vector<float> payload(292);
    for (std::size_t i=0;i<payload.size();++i) payload[i]=float(i)/100;
    const auto path=std::filesystem::temp_directory_path()/
        ("miniinfer17_"+std::to_string(getpid())+".bin");
    auto write=[&](std::size_t count, bool tail=false) {
        std::ofstream f(path,std::ios::binary|std::ios::trunc);
        f.write(reinterpret_cast<const char*>(&c),sizeof(c));
        f.write(reinterpret_cast<const char*>(payload.data()),count*sizeof(float));
        if(tail) f.put('x');
    };
    write(payload.size());
    MLPWeights loaded;
    {
        auto map=std::make_shared<MappedModelFile>(path.string());
        MLPWeightLoader loader(map);
        rejects([&] { loader.load_layer(-1,a); },"negative layer");
        rejects([&] { loader.load_layer(2,a); },"layer upper bound");
        rejects([&] { loader.load_layer(0,nullptr); },"missing load allocator");
        for(int layer=0;layer<2;++layer) {
            loaded=loader.load_layer(layer,a);
            for(const auto& entry : std::vector<std::pair<const Tensor*,std::size_t>>{
                {&loaded.norm,std::size_t(116+4*layer)}, {&loaded.w1,std::size_t(124+28*layer)},
                {&loaded.w2,std::size_t(180+28*layer)}, {&loaded.w3,std::size_t(236+28*layer)}}) {
                std::vector<double> ref(payload.begin()+entry.second,
                                         payload.begin()+entry.second+entry.first->size());
                close(*entry.first,ref,"loaded section",0);
            }
        }
    }
    // 映射已释放，权重仍有自己的存储；真实前向读取它们。
    Tensor hx=tensor({4}), x=on(hx,a,nullptr), y(DataType::FLOAT32,{4},a);
    MLPWorkspace work({4},7,a); MLP mlp(loaded);
    MLPWeights host{loaded.norm.to_cpu(),loaded.w1.to_cpu(),loaded.w2.to_cpu(),loaded.w3.to_cpu()};
    mlp.forward(x,y,work); sync(nullptr,a->device_type()==DeviceType::CUDA);
    close(y,mlp_ref(hx,host,1e-5f).output,"loaded MLP after unmap");
    write(payload.size()-1);
    rejects([&] { MLPWeightLoader short_file(std::make_shared<MappedModelFile>(path.string())); },
            "truncated W3 rejected");
    write(payload.size(),true);
    rejects([&] { MLPWeightLoader unaligned(std::make_shared<MappedModelFile>(path.string())); },
            "partial float payload rejected");
    std::filesystem::remove(path);
}
} // namespace
int main(int argc,char** argv) {
    try {
        bool gpu=false,large=false;
        for(int i=1;i<argc;++i) { gpu|=std::string(argv[i])=="--cuda"; large|=std::string(argv[i])=="--large"; }
        std::shared_ptr<DeviceAllocator> a=cpu;
        if(gpu) a=std::make_shared<CUDADeviceAllocator>();
        swiglu_cases(a,nullptr,large);
        if(!large) { mlp_cases(a,nullptr); loader_cases(a); }
        if(gpu && !large) {
            cudaStream_t s=nullptr;
            check_cuda(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"create stream");
            swiglu_cases(a,s,false); mlp_cases(a,s);
            check_cuda(cudaStreamDestroy(s),"destroy stream");
        }
        std::cout << "PASS " << checks << " checks (" << (gpu?"CUDA":"CPU")
                  << (large?", grid-stride":"") << ")\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
