/* Optional terminal io_uring adapter: one SQE in flight per device. */
#define _GNU_SOURCE
#include "file.h"
#include <liburing.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <unistd.h>

typedef struct {
    if_dev dev;
    int fd, own, failed;
    struct io_uring ring;
    pthread_mutex_t mutex;
} uring_dev;

static int complete(uring_dev *f) {
    int rc;
    do {
        rc = io_uring_submit(&f->ring);
    } while (rc == -EINTR);
    if (rc != 1) {
        /* No SQPOLL: a rejected submission has accepted no operation. Do not
         * submit the queued entry (which borrows a caller buffer) again. */
        f->failed = rc < 0 ? rc : -EIO;
        return f->failed;
    }
    struct io_uring_cqe *cqe = NULL;
    do {
        rc = io_uring_wait_cqe(&f->ring, &cqe);
    } while (rc == -EINTR);
    if (rc < 0) {
        /* Even a failed wait cannot end the buffer borrow. With only one
         * non-IOPOLL operation, the CQ cannot overflow and needs no enter
         * syscall to observe the eventual completion. Preserve the wait error. */
        while (io_uring_peek_cqe(&f->ring, &cqe) != 0)
            sched_yield();
    }
    int result = cqe->res;
    io_uring_cqe_seen(&f->ring, cqe);
    return rc < 0 ? rc : result;
}

static int xfer(uring_dev *f, int write, const if_io *r) {
    char *seg[2] = {r->hdr, r->buf};
    size_t len[2] = {r->hdr_len, r->len};
    size_t total = len[0] + len[1], done = 0;
    while (done < total) {
        struct iovec v[2];
        int count = 0;
        size_t skip = done, room = 0x7ffff000;
        for (int i = 0; i < 2 && room; ++i) {
            if (skip >= len[i]) {
                skip -= len[i];
                continue;
            }
            size_t take = len[i] - skip < room ? len[i] - skip : room;
            v[count++] = (struct iovec){seg[i] + skip, take};
            room -= take;
            skip = 0;
        }
        struct io_uring_sqe *sqe = io_uring_get_sqe(&f->ring);
        if (write)
            io_uring_prep_writev(sqe, f->fd, v, count, r->off + done);
        else
            io_uring_prep_readv(sqe, f->fd, v, count, r->off + done);
        int rc = complete(f);
        if (rc == -EINTR && !f->failed)
            continue;
        if (rc < 0)
            return rc;
        if (!rc)
            return -EIO;
        done += (size_t)rc;
    }
    return 0;
}

static int uring_io(if_dev *d, int write, const if_io *v, size_t n) {
    uring_dev *f = IF_CONTAINER_OF(d, uring_dev, dev);
    int rc = pthread_mutex_lock(&f->mutex);
    if (rc)
        return -rc;
    rc = f->failed;
    for (size_t i = 0; !rc && i < n; ++i)
        rc = xfer(f, write, &v[i]);
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
            struct io_uring_sqe *sqe = io_uring_get_sqe(&f->ring);
            io_uring_prep_fsync(sqe, f->fd, IORING_FSYNC_DATASYNC);
            rc = complete(f);
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

int if_uring_open_fd(int fd, uint64_t size, int own, if_dev **out) {
    uring_dev *f = calloc(1, sizeof *f);
    if (!f)
        return -ENOMEM;
    int rc = pthread_mutex_init(&f->mutex, NULL);
    if (rc) {
        free(f);
        return -rc;
    }
    rc = io_uring_queue_init(8, &f->ring, 0);
    if (rc < 0) {
        pthread_mutex_destroy(&f->mutex);
        free(f);
        return rc;
    }
    f->dev.ops = &uring_ops;
    f->dev.size = size;
    f->fd = fd;
    f->own = own != 0;
    *out = &f->dev;
    return 0;
}
