/* 第 0 层：CRC32C。不分配内存；查找表在首次使用时建好，之后只读。 */
#include "infer/base.h"
#include <sched.h>
#include <stdatomic.h>
#include <string.h>
#if defined(__x86_64__)
#include <nmmintrin.h>
#endif

#define POLY 0x82F63B78u /* Castagnoli 多项式的反射形式 */

/* 只让一个线程建表，其余线程自旋到建好为止。 */
static void once(_Atomic int *state, void (*build)(void)) {
    if (atomic_load_explicit(state, memory_order_acquire) == 2)
        return;
    int idle = 0;
    if (atomic_compare_exchange_strong(state, &idle, 1)) {
        build();
        atomic_store_explicit(state, 2, memory_order_release);
        return;
    }
    while (atomic_load_explicit(state, memory_order_acquire) != 2)
        sched_yield();
}

/* ---- 纯 C：slice-by-8 ---- */
static uint32_t sw_table[8][256];
static _Atomic int sw_state;

static void sw_build(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k)
            c = (c >> 1) ^ (POLY & -(c & 1));
        sw_table[0][i] = c;
    }
    for (uint32_t i = 0; i < 256; ++i)
        for (int t = 1; t < 8; ++t)
            sw_table[t][i] = sw_table[0][sw_table[t - 1][i] & 255] ^ (sw_table[t - 1][i] >> 8);
}

static uint32_t sw_run(uint32_t c, const uint8_t *p, size_t n) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    for (; n >= 8; p += 8, n -= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        uint32_t lo = (uint32_t)v ^ c, hi = (uint32_t)(v >> 32);
        c = sw_table[7][lo & 255] ^ sw_table[6][(lo >> 8) & 255] ^ sw_table[5][(lo >> 16) & 255] ^
            sw_table[4][lo >> 24] ^ sw_table[3][hi & 255] ^ sw_table[2][(hi >> 8) & 255] ^
            sw_table[1][(hi >> 16) & 255] ^ sw_table[0][hi >> 24];
    }
#endif
    while (n--)
        c = sw_table[0][(c ^ *p++) & 255] ^ (c >> 8);
    return c;
}

/* ---- x86-64：SSE4.2 硬件指令，三路并行 ----
 * crc32 指令延迟 3 周期、吞吐 1 周期，单路串行只能跑到 1/3 速度。
 * 把数据切成三段同时算，再用「补 N 个零字节」的线性算子把三路结果合并。 */
#if defined(__x86_64__)
#define BLK 1024                    /* 每路每轮 1 KiB */
static uint32_t hw_shift1[4][256]; /* 寄存器后接 BLK 个零字节 */
static uint32_t hw_shift2[4][256]; /* 寄存器后接 2*BLK 个零字节 */
static _Atomic int hw_state;

static inline uint32_t shift_apply(const uint32_t t[4][256], uint32_t c) {
    return t[0][c & 255] ^ t[1][(c >> 8) & 255] ^ t[2][(c >> 16) & 255] ^ t[3][c >> 24];
}

__attribute__((target("sse4.2"))) static void hw_build(void) {
    for (int k = 0; k < 4; ++k)
        for (uint32_t v = 0; v < 256; ++v) {
            uint64_t r = (uint64_t)v << (8 * k);
            for (size_t i = 0; i < BLK; i += 8)
                r = _mm_crc32_u64(r, 0);
            hw_shift1[k][v] = (uint32_t)r;
        }
    /* 补零算子对寄存器是线性的，所以补 2*BLK 个零就是把 shift1 再用一次。 */
    for (int k = 0; k < 4; ++k)
        for (uint32_t v = 0; v < 256; ++v)
            hw_shift2[k][v] = shift_apply(hw_shift1, hw_shift1[k][v]);
}

__attribute__((target("sse4.2"))) static uint32_t hw_run(uint32_t c, const uint8_t *p, size_t n) {
    uint64_t r = c;
    if (n >= 3 * BLK)
        once(&hw_state, hw_build);
    for (; n >= 3 * BLK; p += 3 * BLK, n -= 3 * BLK) {
        uint64_t a = r, b = 0, d = 0;
        for (size_t i = 0; i < BLK; i += 8) {
            uint64_t x, y, z;
            memcpy(&x, p + i, 8);
            memcpy(&y, p + BLK + i, 8);
            memcpy(&z, p + 2 * BLK + i, 8);
            a = _mm_crc32_u64(a, x);
            b = _mm_crc32_u64(b, y);
            d = _mm_crc32_u64(d, z);
        }
        r = shift_apply(hw_shift2, (uint32_t)a) ^ shift_apply(hw_shift1, (uint32_t)b) ^ (uint32_t)d;
    }
    for (; n >= 8; p += 8, n -= 8) {
        uint64_t x;
        memcpy(&x, p, 8);
        r = _mm_crc32_u64(r, x);
    }
    while (n--)
        r = _mm_crc32_u8((uint32_t)r, *p++);
    return (uint32_t)r;
}
#endif

int if_crc32c_hw(void) {
#if defined(__x86_64__)
    static _Atomic int cached = -1;
    int v = atomic_load_explicit(&cached, memory_order_relaxed);
    if (v < 0) {
        __builtin_cpu_init();
        v = __builtin_cpu_supports("sse4.2") != 0;
        atomic_store_explicit(&cached, v, memory_order_relaxed);
    }
    return v;
#else
    return 0;
#endif
}

uint32_t if_crc32c_sw(uint32_t crc, const void *data, size_t n) {
    once(&sw_state, sw_build);
    return ~sw_run(~crc, data, n);
}

uint32_t if_crc32c(uint32_t crc, const void *data, size_t n) {
#if defined(__x86_64__)
    if (if_crc32c_hw())
        return ~hw_run(~crc, data, n);
#endif
    return if_crc32c_sw(crc, data, n);
}
