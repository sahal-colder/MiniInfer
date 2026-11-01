/* infer/store.h -- 第 3 层：分层 KV 存储。只依赖 base、mem、io。
 *
 * if_store 将逻辑槽 [0, slots) 绑定到共享 TieredStorage 核心的 Block；每槽最多 block_bytes 字节。
 * 数据放在两层：
 *   CPU  最多 cpu_slots 个块，来自 if_pool（cpu_up 可换成 pinned 上游）
 *   SSD  可选，一个 if_dev。核心分配物理记录槽；调用者不得依赖固定偏移或记录内部格式。
 * 策略只有三条：
 *   写    先进 CPU。CPU 满了就把最久没用的槽落到 SSD；已有有效 SSD 副本的直接丢掉 CPU 副本，不重写。
 *   读    CPU 命中就拷贝；否则读 SSD 记录并核对身份和 CRC32C。同一槽 SSD 命中 promote_hits 次后提升回 CPU。
 *   版本  每次提交写递增内容版本，命名替换使用独立 generation；SSD 核对 BlockId、版本、长度和 CRC。
 * SSD 记录只在本进程内有效（主机字节序，文件创建后立刻 unlink），不是持久化格式。
 *
 * 约定：成功返回 0（读返回长度），失败返回 -errno。
 *   -ERANGE   槽号越界，或 len > block_bytes     -EINVAL   参数错（NULL、len 为 0、配置不合法）
 *   -ENOENT   槽里没有数据                        -ENOBUFS  读缓冲小于数据长度
 *   -ENOSPC   CPU 已满且没有 SSD（落盘失败则返回后端的 errno，槽状态不变）
 *   -EBADMSG  SSD 记录的身份或校验不符：缓冲内容不可用，槽状态不变，重写即可修复
 * 批量接口：每个请求独立执行，结果写进各自的 rc，返回值是第一个非零 rc。
 * 线程：策略锁保护槽绑定，核心锁保护 Block；I/O 在锁外由 Lease/操作占用保护。
 * destroy 不能与其他调用并发；AIO 持有 store 时返回 -EBUSY。
 * 上游清理失败时 destroy 保留外壳，仅允许重试 destroy。
 */
#ifndef INFER_STORE_H
#define INFER_STORE_H

#include "infer/io.h"
#include "infer/mem.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IF_AT_CPU 1 /* if_store_where 的返回位 */
#define IF_AT_SSD 2

typedef struct if_store if_store;

typedef struct if_store_cfg {
    size_t block_bytes;        /* 每个槽最多存多少字节 */
    size_t slots;              /* 逻辑槽数 */
    size_t cpu_slots;          /* CPU 最多驻留多少槽，范围 1..slots */
    const if_upstream *cpu_up; /* CPU 块的上游；NULL = 普通内存，换成 pinned 上游就是锁页缓存 */
    if_dev *ssd;               /* 调用者提供的设备：借用，不小于 if_store_dev_bytes，要活得比 store 久 */
    const char *ssd_dir;       /* 或者：在这个目录建私有临时文件，由 store 拥有。两个都为 NULL = 没有 SSD */
    unsigned promote_hits;     /* 同一槽 SSD 命中几次后提升回 CPU；0 = 不提升 */
    size_t piece_bytes;        /* SSD 记录按这个大小分片读取，每片读完立刻校验 CRC；0 = 128 KiB */
} if_store_cfg;

/* 批量请求。写：buf/len 是数据（不会被修改）。读：len 进来是 buf 的容量，成功后改为实际长度。 */
typedef struct if_store_req {
    size_t slot;
    void *buf;
    size_t len;
    int rc; /* 本请求的结果 */
} if_store_req;

typedef struct if_store_stats {
    uint64_t writes, cpu_hits, ssd_hits;
    uint64_t demotions;       /* CPU -> SSD 的落盘次数（写了记录） */
    uint64_t evictions;       /* 丢掉已有 SSD 副本的 CPU 副本（没有 I/O） */
    uint64_t promotions;      /* SSD -> CPU 的提升次数 */
    uint64_t prefetches;      /* 预取读进 CPU 的次数（读命中后的提升不算） */
    uint64_t verify_failures; /* SSD 记录校验失败的次数 */
    uint64_t ssd_written_bytes, ssd_read_bytes;
    size_t populated;  /* 有数据的槽数 */
    size_t cpu_blocks; /* CPU 持有的块数 */
    if_pool_stats pool;
} if_store_stats;

/* SSD 设备需要的字节数；参数非法或溢出返回 0。 */
IF_API uint64_t if_store_dev_bytes(size_t block_bytes, size_t slots);

IF_API int if_store_create(const if_store_cfg *cfg, if_store **out);
IF_API int if_store_destroy(if_store *s);

IF_API int if_store_write(if_store *s, size_t slot, const void *buf, size_t len);
/* 返回读到的长度，或 -errno。 */
IF_API ssize_t if_store_read(if_store *s, size_t slot, void *buf, size_t cap);
IF_API int if_store_write_many(if_store *s, if_store_req *r, size_t n);
IF_API int if_store_read_many(if_store *s, if_store_req *r, size_t n);

/* 槽里没有数据返回 -ENOENT；释放 CPU 块，作废 SSD 记录。 */
IF_API int if_store_erase(if_store *s, size_t slot);
/* 把只在 SSD 上的槽提前读进 CPU，但绝不挤掉别的槽：没有空闲块就什么都不做。已在 CPU 的槽记为最近使用。
 * 返回 1 = 现在在 CPU，0 = 没有空闲块、没读，<0 = -errno（-ENOENT、-ERANGE、-EBADMSG 等，槽状态不变）。 */
IF_API int if_store_prefetch(if_store *s, size_t slot);
/* 返回 IF_AT_CPU / IF_AT_SSD 的组合；槽里没有数据返回 0。 */
IF_API int if_store_where(if_store *s, size_t slot);
IF_API void if_store_get_stats(if_store *s, if_store_stats *out);

#ifdef __cplusplus
}
#endif
#endif
