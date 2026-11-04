/* Device contract: validate the complete batch before entering any backend. */
#include "internal.h"
#include <limits.h>

static int validate(const if_dev *d, const if_io *v, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if ((v[i].hdr_len && !v[i].hdr) || (v[i].len && !v[i].buf))
            return -EINVAL;
        size_t total;
        if (IF_ADD_OVF(v[i].hdr_len, v[i].len, &total) ||
            v[i].off > d->size || total > d->size - v[i].off)
            return -ERANGE;
    }
    return 0;
}

int if_dev_io(if_dev *d, int write, const if_io *v, size_t n) {
    if (!d || (n && !v))
        return -EINVAL;
    int rc = validate(d, v, n);
    if (rc || !n)
        return rc;
    int saved = errno;
    rc = d->ops->io(d, write, v, n);
    errno = saved;
    return rc;
}

int if_dev_close(if_dev *d) {
    if (!d)
        return 0;
    int saved = errno;
    int rc = d->ops->close(d);
    errno = saved;
    return rc;
}

int if_dev_flush(if_dev *d) {
    if (!d)
        return -EINVAL;
    if (!d->ops->flush)
        return -ENOTSUP;
    int saved = errno;
    int rc = d->ops->flush(d);
    errno = saved;
    return rc;
}

static if_file_opener opener(if_io_backend backend) {
    switch (backend) {
    case IF_IO_POSIX:
        return if_posix_open_fd;
#ifdef IF_HAVE_DLSYM
    case IF_IO_DLSYM:
        return if_dlsym_open_fd;
#endif
#ifdef IF_HAVE_URING
    case IF_IO_URING:
        return if_uring_open_fd;
#endif
    default:
        return NULL;
    }
}

int if_dev_open_fd_backend(if_io_backend backend, int fd, uint64_t size, int own, if_dev **out) {
    if (!out)
        return -EINVAL;
    *out = NULL;
    if (fd < 0 || !size || backend < IF_IO_POSIX || backend > IF_IO_URING)
        return -EINVAL;
    if (size > (uint64_t)INT64_MAX)
        return -EFBIG;
    if_file_opener open_fd = opener(backend);
    if (!open_fd)
        return -ENOTSUP;
    int saved = errno;
    int rc = open_fd(fd, size, own, out);
    errno = saved;
    return rc;
}

int if_dev_open_tmp_backend(if_io_backend backend, const char *dir, uint64_t size, if_dev **out) {
    if (!out)
        return -EINVAL;
    *out = NULL;
    if (!dir || !size || backend < IF_IO_POSIX || backend > IF_IO_URING)
        return -EINVAL;
    if (size > (uint64_t)INT64_MAX)
        return -EFBIG;
    if_file_opener open_fd = opener(backend);
    if (!open_fd)
        return -ENOTSUP;
    int saved = errno;
    int rc = if_file_open_tmp(dir, size, open_fd, out);
    errno = saved;
    return rc;
}

int if_dev_open_fd(int fd, uint64_t size, int own, if_dev **out) {
    return if_dev_open_fd_backend(IF_IO_POSIX, fd, size, own, out);
}

int if_dev_open_uring_fd(int fd, uint64_t size, int own, unsigned depth,
                         const struct iovec *buffers, unsigned count, if_dev **out) {
    if (!out)
        return -EINVAL;
    *out = NULL;
    if (fd < 0 || !size || !depth || depth > 64 || count > UINT16_MAX || (count && !buffers))
        return -EINVAL;
    if (size > (uint64_t)INT64_MAX)
        return -EFBIG;
#ifdef IF_HAVE_URING
    int saved = errno;
    int rc = if_uring_open_configured(fd, size, own, depth, buffers, count, out);
    errno = saved;
    return rc;
#else
    (void)own;
    return -ENOTSUP;
#endif
}

int if_dev_open_tmp(const char *dir, uint64_t size, if_dev **out) {
    return if_dev_open_tmp_backend(IF_IO_POSIX, dir, size, out);
}
