#pragma once

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace bench {
using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
inline std::map<std::string, std::string> arguments(int argc, char** argv) {
    if (argc % 2 != 1) throw std::invalid_argument("Arguments must be --key value pairs");
    std::map<std::string, std::string> result;
    for (int i = 1; i < argc; i += 2) {
        if (std::string(argv[i]).rfind("--", 0)) throw std::invalid_argument("Expected --key");
        result.emplace(argv[i] + 2, argv[i + 1]);
    }
    return result;
}
inline std::size_t number(const std::map<std::string, std::string>& args,
                          const std::string& key, std::size_t fallback) {
    const auto entry = args.find(key);
    if (entry == args.end()) return fallback;
    std::size_t end = 0;
    const auto value = std::stoull(entry->second, &end);
    if (entry->second.empty() || entry->second[0] == '-' || end != entry->second.size())
        throw std::invalid_argument("Invalid integer: " + key);
    return value;
}
inline std::string quote(const std::string& value) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { out << '\\' << c; continue; }
        if (c < 32) { out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c); continue; }
        out << c;
    }
    return out.str() + '"';
}
// SplitMix64: the Python allocator driver uses this exact integer trace.
inline std::uint64_t random(std::uint64_t& state) {
    auto value = (state += 0x9e3779b97f4a7c15ULL);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}
inline std::size_t allocation_size(std::uint64_t value) {
    return (std::size_t(256) << (value % 14)) + ((value >> 8) % 16) * 256;
}
}
