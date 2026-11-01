#ifndef INFER_IO_INTERNAL_H
#define INFER_IO_INTERNAL_H
#include "infer/io.h"

typedef int (*if_file_opener)(int fd, uint64_t size, int own, if_dev **out);
int if_file_open_tmp(const char *dir, uint64_t size, if_file_opener open_fd, if_dev **out);
int if_posix_open_fd(int fd, uint64_t size, int own, if_dev **out);
#ifdef IF_HAVE_DLSYM
int if_dlsym_open_fd(int fd, uint64_t size, int own, if_dev **out);
#endif
#ifdef IF_HAVE_URING
int if_uring_open_fd(int fd, uint64_t size, int own, if_dev **out);
#endif
#endif
