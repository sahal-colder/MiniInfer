#include "ssd_record.h"
#include "crc32c/crc32c.h"

#include <cerrno>
#include <climits>
#include <cstring>
#include <system_error>

namespace infer::storage_detail {
namespace {

constexpr unsigned char magic[] = {'M', 'I', 'N', 'I', 'K', 'V', '0', '4'};

// Byte offsets in the little-endian record header shared by v1 files and v2 aligned devices.
constexpr std::size_t format_version_offset = 8;
constexpr std::size_t header_size_offset = 12;
constexpr std::size_t block_id_offset = 16;
constexpr std::size_t content_version_offset = 24;
constexpr std::size_t payload_size_offset = 32;
constexpr std::size_t slot_capacity_offset = 40;
constexpr std::size_t payload_crc_offset = 48;
constexpr std::size_t header_crc_offset = 52;
constexpr std::size_t payload_start_offset = 56;

void put(SSDRecordHeader& header, std::size_t offset, std::uint64_t value, std::size_t bytes) {
    for (std::size_t i = 0; i < bytes; ++i)
        header[offset + i] = static_cast<unsigned char>(value >> (8 * i));
}

std::uint64_t get(const SSDRecordHeader& header, std::size_t offset, std::size_t bytes) {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < bytes; ++i)
        result |= std::uint64_t(header[offset + i]) << (8 * i);
    return result;
}

bool has_aligned_payload(std::size_t payload_offset) {
    return payload_offset != SSDStorageBackend::record_header_bytes;
}

using CRCMatrix = std::array<std::uint32_t, 32>;
std::uint32_t apply_crc(const CRCMatrix& matrix, std::uint32_t value) {
    std::uint32_t result = 0;
    for (std::size_t bit = 0; value; ++bit, value >>= 1)
        if (value & 1) result ^= matrix[bit];
    return result;
}

} // namespace

SSDRecordHeader encode_ssd_header(const BlockMeta& meta, std::size_t capacity, std::uint32_t crc,
                                  std::size_t payload_offset, bool verify) {
    const bool aligned = has_aligned_payload(payload_offset);
    SSDRecordHeader header{};
    std::memcpy(header.data(), magic, sizeof(magic));
    put(header, format_version_offset, aligned ? 2 : 1, 4);
    put(header, header_size_offset, header.size(), 4);
    put(header, block_id_offset, meta.id, 8);
    put(header, content_version_offset, meta.version, 8);
    put(header, payload_size_offset, meta.size, 8);
    put(header, slot_capacity_offset, capacity, 8);
    put(header, payload_crc_offset, crc, 4);
    put(header, payload_start_offset, aligned ? payload_offset : 0, 8);
    if (verify) put(header, header_crc_offset, crc32c(header.data(), header.size()), 4);
    return header;
}

void validate_ssd_header(SSDRecordHeader header, const BlockMeta& expected, std::size_t capacity,
                         std::uint32_t crc, std::size_t payload_offset, bool verify) {
    if (!expected.id || !expected.version || !expected.size || expected.size > capacity)
        throw std::system_error(EBADMSG, std::generic_category(), "Invalid SSD record metadata");

    const bool aligned = has_aligned_payload(payload_offset);
    const auto header_crc = get(header, header_crc_offset, 4);
    put(header, header_crc_offset, 0, 4);
    if (std::memcmp(header.data(), magic, sizeof(magic)) ||
        get(header, format_version_offset, 4) != (aligned ? 2u : 1u) ||
        get(header, header_size_offset, 4) != header.size() ||
        get(header, block_id_offset, 8) != expected.id ||
        get(header, content_version_offset, 8) != expected.version ||
        get(header, payload_size_offset, 8) != expected.size ||
        get(header, slot_capacity_offset, 8) != capacity ||
        get(header, payload_start_offset, 8) != (aligned ? payload_offset : 0) ||
        get(header, payload_crc_offset, 4) != crc ||
        (verify && header_crc != crc32c(header.data(), header.size()))) {
        throw std::system_error(EBADMSG, std::generic_category(),
                                "SSD record header/identity mismatch");
    }
}

void validate_ssd_payload_crc(std::uint32_t crc, const SSDRecordHeader& header) {
    if (crc != get(header, payload_crc_offset, 4))
        throw std::system_error(EBADMSG, std::generic_category(), "SSD payload CRC32C mismatch");
}

// Google CRC32C 接受取反后的值。分段写入沿用原来的累积状态，写入记录头前再取反。
std::uint32_t extend_crc(std::uint32_t crc, const void* data, std::size_t bytes) {
    return ~::crc32c::Extend(~crc, static_cast<const std::uint8_t*>(data), bytes);
}

std::uint32_t crc32c(const void* data, std::size_t bytes) {
    return ::crc32c::Extend(0, static_cast<const std::uint8_t*>(data), bytes);
}

std::uint32_t combine_crc(std::uint32_t first, std::uint32_t second, std::size_t bytes) {
    // Powers of the Castagnoli zero-byte operator combine finalized CRCs
    // without rereading the independently checked payload ranges.
    static const auto powers = [] {
        std::array<CRCMatrix, sizeof(std::size_t) * CHAR_BIT> result{};
        CRCMatrix matrix{};
        matrix[0] = 0x82f63b78u;
        for (std::size_t i = 1; i < matrix.size(); ++i)
            matrix[i] = std::uint32_t(1) << (i - 1);
        for (std::size_t bit = 0; bit < result.size() + 3; ++bit) {
            if (bit >= 3) result[bit - 3] = matrix;
            CRCMatrix square{};
            for (std::size_t i = 0; i < matrix.size(); ++i)
                square[i] = apply_crc(matrix, matrix[i]);
            matrix = square;
        }
        return result;
    }();
    for (std::size_t bit = 0; bytes; ++bit, bytes >>= 1)
        if (bytes & 1) first = apply_crc(powers[bit], first);
    return first ^ second;
}

} // namespace infer::storage_detail
