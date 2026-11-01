#ifndef INFER_IO_FILE_H
#define INFER_IO_FILE_H
#include "internal.h"
#include <sys/uio.h>

typedef struct if_file_calls {
    ssize_t (*readv)(int, const struct iovec *, int, off_t);
    ssize_t (*writev)(int, const struct iovec *, int, off_t);
    int (*sync)(int);
    int (*close)(int);
} if_file_calls;

int if_file_open_fd(int fd, uint64_t size, int own, const if_file_calls *calls, if_dev **out);
#endif
