/* infer/mem.h -- 第 1 层：内存池。只依赖 base，不依赖 io / store。
 *
 * 一个池引擎，上游可换：
 *   host          -- mmap，本层自带（if_upstream_host）
 *   pinned / cuda -- 由上层提供 if_upstream，池代码不变
 * 池自己的元数据（块描述符、页表）直接 mmap，不经过 malloc，也不用 STL，
 * 所以它可以当 malloc 的替身（见 abi 层）。
 *
 * 约定：成功返回 0（或非负结果），失败返回 -errno。
 * 线程：同一个池的函数可以并发调用（一把锁）；destroy 不能与其他调用并发。
 */
#ifndef INFER_MEM_H
#define INFER_MEM_H

#include "infer/base.h"
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 上游合同：
 *   map   返回 4096 对齐、长度 >= bytes 的内存（bytes 总是 4096 的倍数）；成功 0，失败 -errno。
 *   unmap 归还 map 给出的整块（ptr、bytes 与 map 时一致）；失败时内存必须仍然有效。 */
typedef struct if_upstream {
    int (*map)(void *ctx, size_t bytes, void **out);
    int (*unmap)(void *ctx, void *ptr, size_t bytes);
    void *ctx;
} if_upstream;

IF_API const if_upstream *if_upstream_host(void);

typedef struct if_pool_cfg {
    size_t max_reserved; /* 向上游累计持有的字节上限；0 = 不限 */
    size_t max_idle;     /* 完全空闲、留着复用的字节上限；0 = 一空闲就还给上游 */
    size_t small_max;    /* 容量 <= 它的块从 slab 切分（最大 64 KiB）；0 = 全部直连上游 */
} if_pool_cfg;

typedef struct if_pool_stats {
    size_t active_bytes, active_blocks; /* 已分配出去的（按块容量计） */
    size_t idle_bytes;                  /* 完全空闲的 span，可立刻还给上游 */
    size_t reserved_bytes;              /* 向上游持有的总量 */
    uint64_t maps, unmaps, reuses;      /* 上游调用次数；reuses = 不碰上游就满足的分配数 */
} if_pool_stats;

typedef struct if_pool if_pool;

IF_API if_pool_cfg if_pool_cfg_default(void); /* 不限 reserved，idle 64 MiB，small_max 16 KiB */
IF_API size_t if_pool_class_size(size_t n);   /* n 字节实际占用的块容量；过大返回 0 */

/* up、cfg 可为 NULL（host 上游、默认配置）。 */
IF_API int if_pool_create(const if_upstream *up, const if_pool_cfg *cfg, if_pool **out);
/* 还有活跃块返回 -EBUSY，池保持原样。 */
IF_API int if_pool_destroy(if_pool *p);

/* alloc(0) 返回最小块。失败时 *out = NULL。块地址 64 字节对齐。 */
IF_API int if_pool_alloc(if_pool *p, size_t n, void **out);
/* align 为 2 的幂且 <= 4096。 */
IF_API int if_pool_alloc_aligned(if_pool *p, size_t align, size_t n, void **out);
/* free(NULL) 返回 0；不是本池分配、已释放、指向块中间都返回 -EINVAL，且不改变任何状态。
 * 逻辑释放总是生效；返回的其他 -errno 只表示把空闲内存还给上游失败，内存仍留在池里，可再 trim。 */
IF_API int if_pool_free(if_pool *p, void *ptr);
/* 释放 ptr，并告诉调用者它是不是本池的块，用来把「别人的指针」转交出去：
 *   0     已释放（NULL 也返回 0）
 *   1     ptr 不属于本池，什么都没做。大块只登记首页，所以指到大块中间的指针也算不属于
 *   <0    -EINVAL：ptr 属于本池，但不是一个在用块的起点（重复释放、slab 块中间），不改状态；
 *         其他 errno：还上游失败，逻辑释放已生效（同 if_pool_free） */
IF_API int if_pool_try_free(if_pool *p, void *ptr);
/* 给 fork 处理函数用：fork 前 freeze（持有池锁），父子进程里各 thaw 一次。freeze 期间不能调用本池的其他函数。 */
IF_API void if_pool_freeze(if_pool *p);
IF_API void if_pool_thaw(if_pool *p);
/* 活跃块的容量；不是本池活跃块的起始地址返回 -EINVAL（abi 层靠它判断指针归属）。 */
IF_API ssize_t if_pool_usable(if_pool *p, const void *ptr);
/* 把所有完全空闲的 span 还给上游；返回遇到的第一个错误。 */
IF_API int if_pool_trim(if_pool *p);
IF_API void if_pool_get_stats(if_pool *p, if_pool_stats *out);

#ifdef __cplusplus
}
#endif
#endif
