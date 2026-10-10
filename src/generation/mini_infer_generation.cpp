#include "mini_infer/generation/mini_infer_generation.h"

#include <stdexcept>

namespace mini_infer {
GenerationResult generate(LlamaModel& model, const Tokenizer& tokenizer,
                          Sampler& sampler, const std::string& prompt,
                          int32_t max_new_tokens,
                          const std::function<void(const std::string&)>& on_piece) {
    if (max_new_tokens < 0 || tokenizer.vocab_size() != model.config().vocab_size)
        throw std::invalid_argument("invalid generation budget or tokenizer/model mismatch");
    const auto tokens = tokenizer.encode(prompt);
    if (tokens.size() > static_cast<std::size_t>(model.config().seq_len))
        throw std::invalid_argument("prompt exceeds model context (including BOS)");
    GenerationResult result;
    result.prompt_tokens = tokens.size();
    result.stop_reason = "steps";
    model.reset();
    if (max_new_tokens == 0) return result;

    // prefill：每次写入一个 token 的所有层 KV，最后一次 logits 才用于生成。
    const Tensor* logits = nullptr;
    for (int32_t token : tokens) logits = &model.forward(token);
    int32_t previous = tokens.back();
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        const int32_t next = sampler.sample(*logits);
        if (next == Tokenizer::eos_id || next == Tokenizer::bos_id) {
            result.stop_reason = next == Tokenizer::eos_id ? "eos" : "bos";
            break;
        }
        result.generated_tokens.push_back(next);
        if (on_piece) on_piece(tokenizer.decode(previous, next));
        previous = next;
        if (step + 1 == max_new_tokens) break;
        // 最后一格 KV 的 logits 仍能预测一个 token，但不能继续把它送入模型。
        if (model.position() == model.config().seq_len) {
            result.stop_reason = "context";
            break;
        }
        logits = &model.forward(next); // decode：新 token 成为下一轮输入
    }
    return result;
}
}  // namespace mini_infer
