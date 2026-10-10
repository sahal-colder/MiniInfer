#include "mini_infer/generation/mini_infer_tokenizer.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mini_infer {
Tokenizer::Tokenizer(const std::string& path, int32_t vocab_size) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open tokenizer: " + path);
    const auto length = file.tellg();
    if (vocab_size < 259 || length < 4 ||
        static_cast<uint64_t>(length - std::streamoff(4)) < uint64_t(vocab_size) * 9)
        throw std::runtime_error("tokenizer is too short for the model vocabulary");
    file.seekg(0);
    uint64_t remaining = static_cast<uint64_t>(length);
    auto read = [&](void* output, std::size_t count) {
        if (count > remaining || !file.read(static_cast<char*>(output), count))
            throw std::runtime_error("truncated tokenizer");
        remaining -= count;
    };
    uint32_t max_length = 0;
    read(&max_length, sizeof(max_length));
    if (max_length == 0 || max_length > remaining)
        throw std::runtime_error("invalid tokenizer maximum token length");
    pieces_.reserve(vocab_size);
    scores_.reserve(vocab_size);
    for (int32_t id = 0; id < vocab_size; ++id) {
        float score;
        int32_t size;
        read(&score, sizeof(score));
        read(&size, sizeof(size));
        if (!std::isfinite(score) || size <= 0 ||
            static_cast<uint32_t>(size) > max_length || static_cast<uint64_t>(size) > remaining)
            throw std::runtime_error("invalid tokenizer score or piece length");
        std::string piece(size, '\0');
        read(piece.data(), piece.size());
        if (!ids_.emplace(piece, id).second)
            throw std::runtime_error("duplicate tokenizer piece");
        pieces_.push_back(std::move(piece));
        scores_.push_back(score);
    }
    // llama2.c 的导出器也会把 BOS/EOS 文本写成带换行的可读标记；
    // 它们的语义由固定 ID 决定，不能仅接受无换行的拼写。
    const bool bos = pieces_[1] == "<s>" || pieces_[1] == "\n<s>\n";
    const bool eos = pieces_[2] == "</s>" || pieces_[2] == "\n</s>\n";
    if (remaining || pieces_[0] != "<unk>" || !bos || !eos || ids_.count(" ") == 0)
        throw std::runtime_error("unsupported tokenizer or vocabulary size mismatch");
    // byte fallback 不能仅凭 id=byte+3 猜测；确认文件确实采用这个词表布局。
    for (int value = 0; value < 256; ++value) {
        std::ostringstream expected;
        expected << "<0x" << std::uppercase << std::hex << std::setw(2)
                 << std::setfill('0') << value << ">";
        if (pieces_[value + 3] != expected.str())
            throw std::runtime_error("tokenizer does not use Llama byte fallback IDs");
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> tokens{bos_id};
    if (!text.empty()) tokens.push_back(ids_.at(" ")); // SentencePiece dummy prefix
    // 先按 UTF-8 码点查表；词表没有该字符时逐字节回退，中文也能编码。
    for (std::size_t begin = 0; begin < text.size();) {
        const auto first = static_cast<unsigned char>(text[begin]);
        std::size_t width = first < 0x80 ? 1 : (first >= 0xC2 && first < 0xE0 ? 2
                          : (first >= 0xE0 && first < 0xF0 ? 3
                          : (first >= 0xF0 && first < 0xF5 ? 4 : 1)));
        if (begin + width > text.size()) width = 1;
        for (std::size_t j = 1; j < width; ++j) {
            if ((static_cast<unsigned char>(text[begin + j]) & 0xC0) != 0x80) {
                width = 1;
                break;
            }
        }
        const auto found = ids_.find(text.substr(begin, width));
        if (found != ids_.end()) tokens.push_back(found->second);
        else for (std::size_t j = 0; j < width; ++j)
            tokens.push_back(static_cast<unsigned char>(text[begin + j]) + 3);
        begin += width;
    }
    // 每轮选择得分最高的相邻合并，分数相同时保留最左一对。
    // 这是易读的 O(n^2) 版本，适合当前单序列教学入口。
    for (;;) {
        float best_score = -std::numeric_limits<float>::infinity();
        std::size_t best_position = tokens.size();
        int32_t best_id = -1;
        for (std::size_t i = 1; i + 1 < tokens.size(); ++i) {
            const auto found = ids_.find(pieces_[tokens[i]] + pieces_[tokens[i + 1]]);
            if (found != ids_.end() && scores_[found->second] > best_score) {
                best_score = scores_[found->second];
                best_id = found->second;
                best_position = i;
            }
        }
        if (best_id < 0) break;
        tokens[best_position] = best_id;
        tokens.erase(tokens.begin() + best_position + 1);
    }
    return tokens;
}

std::string Tokenizer::decode(int32_t previous_token, int32_t token) const {
    if (token < 0 || token >= vocab_size()) throw std::out_of_range("decode token out of range");
    if (token == bos_id || token == eos_id) return {};
    if (token >= 3 && token < 259) return std::string(1, static_cast<char>(token - 3));
    std::string piece = pieces_[token];
    if (previous_token == bos_id && !piece.empty() && piece.front() == ' ')
        piece.erase(0, 1);
    return piece;
}
}  // namespace mini_infer
