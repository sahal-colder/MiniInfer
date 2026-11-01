/* infer/aio.h -- 第 4 层：store 之上的异步执行器。只依赖 store。
 *
 * 把一批 store 请求交给后台线程，立刻拿到一个代号（token），之后用 wait / poll 取结果。
 * 调用方可以在 SSD 读写后台进行时去做别的事，比如启动 GPU 拷贝。
 *   操作    IF_AIO_WRITE / IF_AIO_READ 等同 if_store_write_many / if_store_read_many；
 *           IF_AIO_PREFETCH 对每个槽调用 if_store_prefetch，请求里只用 slot。
 *   顺序    严格先进先出：后提交的作业在前一个完成后才开始，同槽、同缓冲的先后依赖因此天然成立，没有冲突检查。
 *           现在只有一个工作线程：store 的 I/O 在锁里，多个线程也不会并行。store 拆锁后再加线程和冲突检查。
 *   容量    最多 IF_AIO_MAX_JOBS 个作业在册，每个最多 IF_AIO_MAX_REQ 条请求。作业完成后仍占位置，直到被
 *           wait / poll / drain 取走；满了返回 -EBUSY，不阻塞。
 *   所有权  请求数组在 submit 时复制，调用方可以立刻复用；数据缓冲不复制，要保持有效直到作业被取走。
 * 约定：成功返回 >= 0，失败返回 -errno。
 *   -EINVAL  参数错（NULL、n 为 0 或超过上限、未知操作）     -EBUSY  作业位置已满
 *   -ENOENT  代号不存在或已被取走
 * wait / poll / drain 的返回值是作业结果：第一个小于 0 的请求结果，没有则为 0。poll 在作业没完成时返回 1。
 * out 不为 NULL 时收到该作业 n 条请求的最终 slot / len / rc（预取的 rc：1 = 已在 CPU，0 = 没有空闲块而跳过）。
 * drain 等所有在册作业做完并取走，返回其中第一个失败；调用方保证此时没有并发的 submit 和 wait。
 * destroy 先让已接受的作业全部做完；它不能与其他调用并发。aio 不拥有 store，store 要活得比 aio 久。
 */
#ifndef INFER_AIO_H
#define INFER_AIO_H

#include "infer/store.h"

#ifdef __cplusplus
extern "C" {
#endif

#define IF_AIO_MAX_JOBS 16
#define IF_AIO_MAX_REQ 64

enum { IF_AIO_WRITE = 0, IF_AIO_READ = 1, IF_AIO_PREFETCH = 2 };

typedef struct if_aio if_aio;

typedef struct if_aio_stats {
    uint64_t submitted, completed, rejected; /* rejected = 因作业位置已满被拒的次数 */
    size_t inflight, peak;                   /* 在册作业数（含已完成但未取走的）及其峰值 */
} if_aio_stats;

IF_API int if_aio_create(if_store *s, if_aio **out);
IF_API int if_aio_destroy(if_aio *a);

IF_API int if_aio_submit(if_aio *a, int op, const if_store_req *r, size_t n, uint64_t *token);
/* 阻塞到作业完成并取走。out 要能放下提交时的 n 条请求。 */
IF_API int if_aio_wait(if_aio *a, uint64_t token, if_store_req *out);
/* 不阻塞：没完成返回 1，否则同 wait。 */
IF_API int if_aio_poll(if_aio *a, uint64_t token, if_store_req *out);
IF_API int if_aio_drain(if_aio *a);
IF_API void if_aio_get_stats(if_aio *a, if_aio_stats *out);

#ifdef __cplusplus
}
#endif
#endif
