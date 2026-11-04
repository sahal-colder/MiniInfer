/* Bounded positional I/O. Every submitted operation relinquishes its buffers
 * before the adapter returns, including when another SQE in its batch fails. */
#define _GNU_SOURCE
#include "file.h"
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { max_depth = 64, sector_bytes = 4096, max_io_bytes = 0x7ffff000 };

typedef struct {
    if_dev dev;
    int fd, own, failed;
    unsigned depth, fixed_count;
    struct io_uring ring;
    pthread_mutex_t mutex;
    struct iovec fixed[];
} uring_dev;

typedef struct {
    const if_io *request;
    size_t total, done, bytes;
    struct iovec segments[2];
    int result;
} transfer;

static int complete(uring_dev *f, transfer *pending, unsigned count) {
    unsigned submitted = 0;
    while (submitted < count) {
        int rc = io_uring_submit(&f->ring);
        if (rc == -EINTR)
            continue;
        if (rc <= 0) {
            /* No SQPOLL. A failed enter leaves unaccepted entries queued.
             * Disable future submissions so those entries cannot borrow freed memory. */
            f->failed = rc < 0 ? rc : -EIO;
            break;
        }
        submitted += (unsigned)rc;
    }
    int error = f->failed;
    for (unsigned i = 0; i < submitted; ++i) {
        struct io_uring_cqe *cqe = NULL;
        int rc;
        do {
            rc = io_uring_wait_cqe(&f->ring, &cqe);
        } while (rc == -EINTR);
        if (rc < 0) {
            if (!error)
                error = rc;
            /* The CQ fits the entire wave and IOPOLL is disabled. */
            while (io_uring_peek_cqe(&f->ring, &cqe) != 0)
                sched_yield();
        }
        pending[cqe->user_data].result = cqe->res;
        io_uring_cqe_seen(&f->ring, cqe);
    }
    return error;
}

static int fixed_index(const uring_dev *f, const struct iovec *segment) {
    const uintptr_t addr = (uintptr_t)segment->iov_base;
    for (unsigned i = 0; i < f->fixed_count; ++i) {
        const uintptr_t base = (uintptr_t)f->fixed[i].iov_base;
        if (addr >= base && addr - base < f->fixed[i].iov_len &&
            segment->iov_len <= f->fixed[i].iov_len - (addr - base))
            return (int)i;
    }
    return -1;
}

static void prepare(uring_dev *f, int write, transfer *part, unsigned index) {
    const if_io *r = part->request;
    char *seg[2] = {r->hdr, r->buf};
    size_t len[2] = {r->hdr_len, r->len};
    unsigned count = 0;
    size_t skip = part->done, room = max_io_bytes;
    for (int i = 0; i < 2 && room; ++i) {
        if (skip >= len[i]) {
            skip -= len[i];
            continue;
        }
        const size_t take = len[i] - skip < room ? len[i] - skip : room;
        part->segments[count++] = (struct iovec){seg[i] + skip, take};
        room -= take;
        skip = 0;
    }
    part->bytes = max_io_bytes - room;
    const uint64_t offset = r->off + part->done;
    struct io_uring_sqe *sqe = io_uring_get_sqe(&f->ring);
    const int fixed = count == 1 ? fixed_index(f, part->segments) : -1;
    if (fixed >= 0) {
        if (write)
            io_uring_prep_write_fixed(sqe, f->fd, part->segments[0].iov_base, part->bytes, offset, fixed);
        else
            io_uring_prep_read_fixed(sqe, f->fd, part->segments[0].iov_base, part->bytes, offset, fixed);
    } else if (write)
        io_uring_prep_writev(sqe, f->fd, part->segments, count, offset);
    else
        io_uring_prep_readv(sqe, f->fd, part->segments, count, offset);
    sqe->user_data = index;
}

static int uring_io(if_dev *d, int write, const if_io *v, size_t n) {
    uring_dev *f = IF_CONTAINER_OF(d, uring_dev, dev);
    int rc = pthread_mutex_lock(&f->mutex);
    if (rc)
        return -rc;
    rc = f->failed;
    transfer pending[max_depth];
    unsigned active = 0;
    size_t next = 0;
    while (!rc && (active || next < n)) {
        while (active < f->depth && next < n) {
            const if_io *r = &v[next++];
            if (r->hdr_len || r->len)
                pending[active++] = (transfer){.request = r, .total = r->hdr_len + r->len};
        }
        if (!active)
            break;
        for (unsigned i = 0; i < active; ++i)
            prepare(f, write, &pending[i], i);
        rc = complete(f, pending, active);
        unsigned remaining = 0;
        for (unsigned i = 0; !rc && i < active; ++i) {
            transfer *part = &pending[i];
            if (part->result != -EINTR) {
                if (part->result < 0)
                    rc = part->result;
                else if (!part->result || (size_t)part->result > part->bytes)
                    rc = -EIO; /* EOF/no progress or invalid completion. */
                else
                    part->done += (size_t)part->result;
            }
            if (part->done < part->total)
                pending[remaining++] = *part;
        }
        active = remaining;
    }
    pthread_mutex_unlock(&f->mutex);
    return rc;
}

static int uring_flush(if_dev *d) {
    uring_dev *f = IF_CONTAINER_OF(d, uring_dev, dev);
    int rc = pthread_mutex_lock(&f->mutex);
    if (rc)
        return -rc;
    rc = f->failed;
    if (!rc) {
        do {
            transfer pending;
            struct io_uring_sqe *sqe = io_uring_get_sqe(&f->ring);
            io_uring_prep_fsync(sqe, f->fd, IORING_FSYNC_DATASYNC);
            sqe->user_data = 0;
            rc = complete(f, &pending, 1);
            if (!rc)
                rc = pending.result;
        } while (rc == -EINTR && !f->failed);
    }
    pthread_mutex_unlock(&f->mutex);
    return rc;
}

static int uring_close(if_dev *d) {
    uring_dev *f = IF_CONTAINER_OF(d, uring_dev, dev);
    io_uring_queue_exit(&f->ring);
    pthread_mutex_destroy(&f->mutex);
    int rc = f->own && close(f->fd) ? -errno : 0;
    free(f);
    return rc;
}

static const if_dev_ops uring_ops = {uring_io, uring_close, uring_flush};

int if_uring_open_configured(int fd, uint64_t size, int own, unsigned depth,
                             const struct iovec *buffers, unsigned count, if_dev **out) {
    for (unsigned i = 0; i < count; ++i)
        if (!buffers[i].iov_base || !buffers[i].iov_len ||
            (uintptr_t)buffers[i].iov_base % sector_bytes || buffers[i].iov_len % sector_bytes)
            return -EINVAL;
    uring_dev *f = calloc(1, sizeof *f + count * sizeof *buffers);
    if (!f)
        return -ENOMEM;
    int rc = pthread_mutex_init(&f->mutex, NULL);
    if (rc) {
        free(f);
        return -rc;
    }
    rc = io_uring_queue_init(depth, &f->ring, 0);
    if (!rc && count) {
        rc = io_uring_register_buffers(&f->ring, buffers, count);
        if (rc < 0)
            io_uring_queue_exit(&f->ring);
        else
            memcpy(f->fixed, buffers, count * sizeof *buffers);
    }
    if (rc < 0) {
        pthread_mutex_destroy(&f->mutex);
        free(f);
        return rc;
    }
    f->dev.ops = &uring_ops;
    f->dev.size = size;
    f->fd = fd;
    f->own = own != 0;
    f->depth = depth;
    f->fixed_count = count;
    *out = &f->dev;
    return 0;
}

int if_uring_open_fd(int fd, uint64_t size, int own, if_dev **out) {
    return if_uring_open_configured(fd, size, own, 1, NULL, 0, out);
}
