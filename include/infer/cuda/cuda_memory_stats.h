// 保留显存池对外返回的全部统计字段。
#pragma once

#include <cstddef>

namespace infer {

// 字段顺序、类型和含义保持原 public API 不变。
// reserved = active_capacity + pending + ready + graph_reserved。
// segments + splits - merges = active_blocks + pending_blocks + ready_blocks。
struct CUDAMemoryPoolStats {
    std::size_t active_requested_bytes = 0;
    std::size_t active_capacity_bytes = 0;
    std::size_t pending_bytes = 0;
    std::size_t ready_bytes = 0;
    std::size_t reserved_bytes = 0;
    std::size_t peak_reserved_bytes = 0;

    std::size_t active_blocks = 0;
    std::size_t pending_blocks = 0;
    std::size_t ready_blocks = 0;

    std::size_t upstream_allocations = 0;
    std::size_t upstream_releases = 0;
    std::size_t reuse_count = 0;

    std::size_t event_records = 0;
    std::size_t event_queries = 0;
    std::size_t completion_waits = 0;
    std::size_t oom_retries = 0;

    std::size_t segments = 0;
    std::size_t splits = 0;
    std::size_t merges = 0;

    std::size_t graph_reserved_bytes = 0;
    std::size_t graph_peak_used_bytes = 0;
    std::size_t graph_count = 0;
};

} // namespace infer
