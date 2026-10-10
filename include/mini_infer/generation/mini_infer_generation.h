#pragma once

#include "mini_infer/model/mini_infer_model.h"
#include "mini_infer/generation/mini_infer_sampler.h"
#include "mini_infer/generation/mini_infer_tokenizer.h"
#include <functional>

namespace mini_infer {
struct GenerationResult {
    std::size_t prompt_tokens = 0;
    std::vector<int32_t> generated_tokens; // 不包含终止符 BOS/EOS
    std::string stop_reason;              // eos / bos / context / steps
};

// 一次完整请求：重置旧缓存 -> prefill -> 自回归 decode。
// max_new_tokens 只计算新 token，不把 prompt 长度算入生成预算。
// 回调每次收到一个解码片段（UTF-8 可能跨多个片段），方便流式打印。
GenerationResult generate(LlamaModel& model, const Tokenizer& tokenizer,
                          Sampler& sampler, const std::string& prompt,
                          int32_t max_new_tokens,
                          const std::function<void(const std::string&)>& on_piece = {});
}  // namespace mini_infer
