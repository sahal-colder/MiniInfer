/* Optional Linux adapter. Only this translation unit depends on libdl. */
#define _GNU_SOURCE
#include "file.h"
#include <dlfcn.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>

static pthread_once_t once = PTHREAD_ONCE_INIT;
static _Thread_local int resolving;
static if_file_calls calls;
static int ready;

static ssize_t syscall_readv(int fd, const struct iovec *v, int n, off_t off) {
#if __SIZEOF_LONG__ == 8
    return syscall(SYS_preadv, fd, v, n, (unsigned long)off, 0UL);
#else
    return syscall(SYS_preadv, fd, v, n, (unsigned long)off, (unsigned long)((uint64_t)off >> 32));
#endif
}

static ssize_t syscall_writev(int fd, const struct iovec *v, int n, off_t off) {
#if __SIZEOF_LONG__ == 8
    return syscall(SYS_pwritev, fd, v, n, (unsigned long)off, 0UL);
#else
    return syscall(SYS_pwritev, fd, v, n, (unsigned long)off, (unsigned long)((uint64_t)off >> 32));
#endif
}

static int syscall_sync(int fd) { return (int)syscall(SYS_fdatasync, fd); }
static int syscall_close(int fd) { return (int)syscall(SYS_close, fd); }

static void resolve(void) {
    calls.readv = (ssize_t (*)(int, const struct iovec *, int, off_t))dlsym(RTLD_NEXT, "preadv");
    calls.writev = (ssize_t (*)(int, const struct iovec *, int, off_t))dlsym(RTLD_NEXT, "pwritev");
    calls.sync = (int (*)(int))dlsym(RTLD_NEXT, "fdatasync");
    calls.close = (int (*)(int))dlsym(RTLD_NEXT, "close");
    ready = calls.readv && calls.writev && calls.sync && calls.close;
}

int if_dlsym_open_fd(int fd, uint64_t size, int own, if_dev **out) {
    /* dlsym can allocate through a preload shim which re-enters this adapter.
     * Never re-enter pthread_once on the resolving thread. */
    static const if_file_calls bootstrap = {syscall_readv, syscall_writev, syscall_sync, syscall_close};
    if (resolving)
        return if_file_open_fd(fd, size, own, &bootstrap, out);
    resolving = 1;
    int rc = pthread_once(&once, resolve);
    resolving = 0;
    if (rc)
        return -rc;
    if (!ready)
        return -ENOTSUP;
    return if_file_open_fd(fd, size, own, &calls, out);
}
