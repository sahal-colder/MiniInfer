#define _GNU_SOURCE
#include "file.h"
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define MAX_CHUNK ((size_t)0x7ffff000)

typedef struct {
    if_dev dev;
    int fd, own;
    const if_file_calls *calls;
} file_dev;

/* Continue short transfers across both segments; zero progress is terminal. */
static int xfer(file_dev *f, int write, const if_io *r) {
    char *seg[2] = {r->hdr, r->buf};
    size_t len[2] = {r->hdr_len, r->len};
    size_t total = len[0] + len[1], done = 0;
    while (done < total) {
        struct iovec v[2];
        int cnt = 0;
        size_t skip = done, room = MAX_CHUNK;
        for (int i = 0; i < 2 && room; ++i) {
            if (skip >= len[i]) {
                skip -= len[i];
                continue;
            }
            size_t take = len[i] - skip < room ? len[i] - skip : room;
            v[cnt++] = (struct iovec){seg[i] + skip, take};
            room -= take;
            skip = 0;
        }
        off_t at = (off_t)(r->off + done);
        ssize_t k = write ? f->calls->writev(f->fd, v, cnt, at) : f->calls->readv(f->fd, v, cnt, at);
        if (k < 0) {
            if (errno == EINTR)
                continue;
            return -errno;
        }
        if (!k)
            return -EIO;
        done += (size_t)k;
    }
    return 0;
}

static int file_io(if_dev *d, int write, const if_io *v, size_t n) {
    file_dev *f = IF_CONTAINER_OF(d, file_dev, dev);
    for (size_t i = 0; i < n; ++i) {
        int rc = xfer(f, write, &v[i]);
        if (rc)
            return rc;
    }
    return 0;
}

static int file_close(if_dev *d) {
    file_dev *f = IF_CONTAINER_OF(d, file_dev, dev);
    int rc = (f->own && f->calls->close(f->fd)) ? -errno : 0;
    free(f);
    return rc;
}

static int file_flush(if_dev *d) {
    file_dev *f = IF_CONTAINER_OF(d, file_dev, dev);
    int rc;
    do {
        rc = f->calls->sync(f->fd) ? -errno : 0;
    } while (rc == -EINTR);
    return rc;
}

static const if_dev_ops file_ops = {file_io, file_close, file_flush};

int if_file_open_fd(int fd, uint64_t size, int own, const if_file_calls *calls, if_dev **out) {
    file_dev *f = calloc(1, sizeof *f);
    if (!f)
        return -ENOMEM;
    f->dev.ops = &file_ops;
    f->dev.size = size;
    f->fd = fd;
    f->own = own != 0;
    f->calls = calls;
    *out = &f->dev;
    return 0;
}

int if_posix_open_fd(int fd, uint64_t size, int own, if_dev **out) {
    static const if_file_calls calls = {preadv, pwritev, fdatasync, close};
    return if_file_open_fd(fd, size, own, &calls, out);
}

int if_file_open_tmp(const char *dir, uint64_t size, if_file_opener open_fd, if_dev **out) {
    char name[PATH_MAX];
    int n = snprintf(name, sizeof name, "%s/infer-io-XXXXXX", dir);
    if (n < 0 || (size_t)n >= sizeof name)
        return -ENAMETOOLONG;
    int fd = mkostemp(name, O_CLOEXEC);
    if (fd < 0)
        return -errno;
    int rc;
    do {
        rc = ftruncate(fd, (off_t)size) ? -errno : 0;
    } while (rc == -EINTR);
    if (unlink(name) && !rc)
        rc = -errno;
    if (!rc)
        rc = open_fd(fd, size, 1, out);
    if (rc)
        close(fd);
    return rc;
}
