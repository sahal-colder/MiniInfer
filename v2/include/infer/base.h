/* infer/base.h -- 第 0 层：不分配内存、不做 I/O、不创建线程，只放所有层共用的小工具。
 *
 * 全库约定：函数成功返回 >= 0，失败返回 -errno。不抛异常，也不靠全局 errno 传错。
 * 依赖方向：base <- mem <- io <- store <- abi。上层可以用下层，反过来不行。
 */
#ifndef INFER_BASE_H
#define INFER_BASE_H

#include <errno.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IF_API __attribute__((visibility("default")))
#define IF_LIKELY(x) __builtin_expect(!!(x), 1)
#define IF_UNLIKELY(x) __builtin_expect(!!(x), 0)
#define IF_ALIGN_UP(x, a) ((((x) + (a)-1)) & ~(__typeof__((x) + (a)))((a)-1))
#define IF_IS_POW2(x) ((x) && !((x) & ((x)-1)))
#define IF_CONTAINER_OF(p, T, m) ((T *)((char *)(p)-offsetof(T, m)))

/* 溢出时返回非零：if (IF_ADD_OVF(a, b, &r)) return -EOVERFLOW; */
#define IF_ADD_OVF(a, b, out) __builtin_add_overflow((a), (b), (out))
#define IF_MUL_OVF(a, b, out) __builtin_mul_overflow((a), (b), (out))

/* 侵入式双向循环链表：节点嵌在调用者的结构里，增删不分配内存。 */
typedef struct if_node {
    struct if_node *prev, *next;
} if_node;

static inline void if_list_init(if_node *head) { head->prev = head->next = head; }
static inline int if_list_empty(const if_node *head) { return head->next == head; }
static inline void if_list_add(if_node *head, if_node *n) { /* 插到表头 */
    n->next = head->next;
    n->prev = head;
    head->next->prev = n;
    head->next = n;
}
static inline void if_list_add_tail(if_node *head, if_node *n) {
    n->prev = head->prev;
    n->next = head;
    head->prev->next = n;
    head->prev = n;
}
static inline void if_list_del(if_node *n) {
    n->prev->next = n->next;
    n->next->prev = n->prev;
    n->prev = n->next = n;
}

/* 位图：给分桶用。if_bit_next 找下标 >= from 的第一个置位，找不到返回 -1。 */
static inline void if_bit_set(uint64_t *w, unsigned i) { w[i >> 6] |= 1ull << (i & 63); }
static inline void if_bit_clear(uint64_t *w, unsigned i) { w[i >> 6] &= ~(1ull << (i & 63)); }
static inline int if_bit_test(const uint64_t *w, unsigned i) { return (int)((w[i >> 6] >> (i & 63)) & 1); }
static inline int if_bit_next(const uint64_t *w, unsigned nbits, unsigned from) {
    if (from >= nbits)
        return -1;
    unsigned i = from >> 6, words = (nbits + 63) >> 6;
    uint64_t x = w[i] & (~0ull << (from & 63));
    while (!x) {
        if (++i == words)
            return -1;
        x = w[i];
    }
    unsigned b = (i << 6) + (unsigned)__builtin_ctzll(x);
    return b < nbits ? (int)b : -1;
}

/* CRC32C (Castagnoli)。crc 传 0 开始；把上一次的返回值传回来可以分段续算。
 * if_crc32c 在 x86-64 上运行时选 SSE4.2 硬件指令（三路并行），否则用纯 C 查表。 */
IF_API uint32_t if_crc32c(uint32_t crc, const void *data, size_t n);
IF_API uint32_t if_crc32c_sw(uint32_t crc, const void *data, size_t n); /* 纯 C，供对拍 */
IF_API int if_crc32c_hw(void);                                          /* 1 表示 if_crc32c 走硬件 */

#ifdef __cplusplus
}
#endif
#endif
