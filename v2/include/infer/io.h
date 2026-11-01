/* infer/io.h -- 第 2 层：设备 I/O。只依赖 base，不依赖 mem。
 *
 * 设备是一段 [0, size) 的字节空间。I/O 是一批定位读写，没有文件游标，所以不同线程可以
 * 并发使用同一个设备。后端通过 if_dev_ops 接入：本层自带 POSIX 文件后端，别人可以不改本层
 * 代码另加 io_uring、SPDK 或内存设备（测试里就有一个内存后端）。
 *
 * 约定：成功返回 0，失败返回 -errno。
 *   -EINVAL  参数错：NULL，或缓冲为 NULL 而长度非零
 *   -ERANGE  请求超出设备范围（含算术溢出）；整批请求一个都不执行
 *   其他     来自后端；批内按顺序执行，遇到第一个错误就停止，已完成的请求保持完成
 * 公共入口保留调用者 errno，以返回值传错。
 */
#ifndef INFER_IO_H
#define INFER_IO_H

#include "infer/base.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 一个请求覆盖设备上连续的 hdr_len + len 字节：先 hdr 段，后 buf 段，从 off 开始。
 * 记录头和载荷在内存里是两块时，一次系统调用写出，不用拼接拷贝。任何一段可以为空。
 * 写操作不会修改 hdr / buf 的内容。 */
typedef struct if_io {
    void *hdr;
    size_t hdr_len;
    void *buf;
    size_t len;
    uint64_t off;
} if_io;

typedef struct if_dev if_dev;

/* 后端合同：
 *   io    收到的请求已经校验过（指针、范围），后端只管执行；返回时所有已接受的 I/O 都已结束。
 *   close 释放设备对象本身（包括 if_dev 所在的那块内存）。
 *   flush 可选，等待已完成写入持久化；NULL 表示不支持。 */
typedef struct if_dev_ops {
    int (*io)(if_dev *d, int write, const if_io *v, size_t n);
    int (*close)(if_dev *d);
    int (*flush)(if_dev *d);
} if_dev_ops;

/* 后端把 if_dev 放在自己结构体的第一个成员。 */
struct if_dev {
    const if_dev_ops *ops;
    uint64_t size;
};

typedef enum if_io_backend {
    IF_IO_POSIX,
    IF_IO_DLSYM,
    IF_IO_URING
} if_io_backend;

/* Explicit selection never falls back to another backend. A backend omitted
 * from the build returns -ENOTSUP. Open failure leaves *out NULL and fd owned
 * by the caller. Legacy open functions select POSIX. */
IF_API int if_dev_open_tmp_backend(if_io_backend backend, const char *dir, uint64_t size, if_dev **out);
IF_API int if_dev_open_fd_backend(if_io_backend backend, int fd, uint64_t size, int own, if_dev **out);

/* 私有临时文件：建在 dir 下，立刻 unlink，关闭或进程退出后自动消失；稀疏，未写区域读出为零。
 * size 必须在 1..INT64_MAX。失败时 *out = NULL。 */
IF_API int if_dev_open_tmp(const char *dir, uint64_t size, if_dev **out);
/* 包装一个已打开的 fd。own 非零则 close 时关闭 fd；失败时 fd 仍归调用者。
 * size 是声明的设备大小，不要求文件已经有这么长（读到文件末尾得到 -EIO）。 */
IF_API int if_dev_open_fd(int fd, uint64_t size, int own, if_dev **out);
/* close(NULL) 返回 0。不能与其他调用并发。 */
IF_API int if_dev_close(if_dev *d);
/* Persist completed writes (fdatasync for file backends). An omitted backend
 * callback returns -ENOTSUP; flush(NULL) returns -EINVAL. */
IF_API int if_dev_flush(if_dev *d);

/* 校验后交给后端。n 可以为 0。 */
IF_API int if_dev_io(if_dev *d, int write, const if_io *v, size_t n);
static inline int if_dev_read(if_dev *d, const if_io *v, size_t n) { return if_dev_io(d, 0, v, n); }
static inline int if_dev_write(if_dev *d, const if_io *v, size_t n) { return if_dev_io(d, 1, v, n); }

#ifdef __cplusplus
}
#endif
#endif
