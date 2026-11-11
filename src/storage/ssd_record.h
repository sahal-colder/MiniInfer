#pragma once

#include "infer/storage/storage_backend.h"

#include <array>

namespace infer::storage_detail {

using SSDRecordHeader = std::array<unsigned char, SSDStorageBackend::record_header_bytes>;

SSDRecordHeader encode_ssd_header(const BlockMeta& meta, std::size_t capacity, std::uint32_t crc,
                                  std::size_t payload_offset, bool verify);
void validate_ssd_header(SSDRecordHeader header, const BlockMeta& expected, std::size_t capacity,
                         std::uint32_t crc, std::size_t payload_offset, bool verify);
void validate_ssd_payload_crc(std::uint32_t crc, const SSDRecordHeader& header);
std::uint32_t extend_crc(std::uint32_t crc, const void* data, std::size_t bytes);
std::uint32_t combine_crc(std::uint32_t first, std::uint32_t second, std::size_t bytes);

} // namespace infer::storage_detail
