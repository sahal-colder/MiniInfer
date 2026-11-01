#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace mini_infer {
// CPU 字符串处理，不需要 CUDA。读取 llama2.c tokenizer.bin，而不是
// SentencePiece 的 tokenizer.model 或 Hugging Face 的 tokenizer.json。
// 文件：uint32 max_length，随后每个词条为 float score + int32 length + bytes。
// 词表必须与模型配套；ID 0/1/2 = UNK/BOS/EOS，ID 3..258 = 原始字节。
class Tokenizer {
public:
    static constexpr int32_t bos_id = 1;
    static constexpr int32_t eos_id = 2;
    Tokenizer(const std::string& path, int32_t vocab_size);
    std::vector<int32_t> encode(const std::string& text) const; // 自动加 BOS
    std::string decode(int32_t previous_token, int32_t token) const;
    int32_t vocab_size() const noexcept { return static_cast<int32_t>(pieces_.size()); }
private:
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::unordered_map<std::string, int32_t> ids_;
};
}  // namespace mini_infer
