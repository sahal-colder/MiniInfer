/* 第 1 层：内存池引擎。
 *
 *   - 块按容量分桶：64B 一档到 1KiB，之后每个 2 的幂分 4 档，浪费 < 25%。
 *   - 小块（容量 <= small_max）：向上游要 64 倍容量的 slab，切成 64 个槽，空闲状态是
 *     一个 64 位位图。大块：一整块上游内存就是一个块。
 *   - 一个 span 描述符覆盖这两种情况；每个桶有一条「还有空闲」的 span 链，分配 O(1)。
 *   - 指针 -> span 靠三级页表（4KiB 粒度），不用哈希表；slab 登记全部页，大块只登记首页。
 *   - 元数据来自 arena（直接 mmap），本层不调用 malloc。
 */
#include "infer/mem.h"
#include <pthread.h>
#include <string.h>
#include <sys/mman.h>

#define PAGE ((size_t)4096)
#define PAGE_SHIFT 12
#define BINS 160u
#define MAX_BYTES ((size_t)1 << 46)
#define SLOTS 64u
#define FULL (~(uint64_t)0)
#define PM_BITS 12
#define PM_MASK ((1u << PM_BITS) - 1)
#define PM_PAGES ((uint64_t)1 << (3 * PM_BITS))
#define ARENA_MIN ((size_t)1 << 20)
#define ARENA_HDR ((size_t)64)
#define SMALL_LIMIT ((size_t)65536)

/* ---- 上游：host（mmap） ---- */
static int host_map(void *ctx, size_t bytes, void **out) {
    (void)ctx;
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED)
        return -errno;
    *out = p;
    return 0;
}
static int host_unmap(void *ctx, void *ptr, size_t bytes) {
    (void)ctx;
    return munmap(ptr, bytes) ? -errno : 0;
}
static const if_upstream host_upstream = {host_map, host_unmap, NULL};
const if_upstream *if_upstream_host(void) { return &host_upstream; }

/* ---- arena：元数据只增不减的 mmap 堆，永远不依赖 libc malloc ---- */
typedef struct arena_chunk {
    struct arena_chunk *next;
    size_t bytes;
} arena_chunk;

typedef struct {
    arena_chunk *chunks;
    char *cur, *end;
} arena;

static int arena_alloc(arena *a, size_t n, void **out) {
    n = IF_ALIGN_UP(n, (size_t)16);
    if (!a->cur || (size_t)(a->end - a->cur) < n) {
        size_t bytes = IF_ALIGN_UP(n + ARENA_HDR, PAGE);
        if (bytes < ARENA_MIN)
            bytes = ARENA_MIN;
        void *mem = NULL;
        int rc = host_map(NULL, bytes, &mem);
        if (rc)
            return rc;
        arena_chunk *c = mem;
        c->next = a->chunks;
        c->bytes = bytes;
        a->chunks = c;
        a->cur = (char *)mem + ARENA_HDR;
        a->end = (char *)mem + bytes;
    }
    *out = a->cur;
    a->cur += n;
    return 0;
}

/* ---- 分桶：容量表和反查 ---- */
static size_t cap_of(unsigned bin) {
    if (bin < 16)
        return (size_t)(bin + 1) * 64;
    unsigned e = 10 + (bin - 16) / 4, part = (bin - 16) % 4 + 1;
    size_t base = (size_t)1 << e;
    return base + base / 4 * part;
}

/* n 必须在 [1, MAX_BYTES]。 */
static unsigned bin_of(size_t n) {
    if (n <= 1024)
        return (unsigned)((n + 63) / 64) - 1;
    unsigned e = 63u - (unsigned)__builtin_clzll(n - 1);
    size_t base = (size_t)1 << e, q = base / 4;
    return 16 + 4 * (e - 10) + (unsigned)((n - base + q - 1) / q) - 1;
}

size_t if_pool_class_size(size_t n) {
    return n > MAX_BYTES ? 0 : cap_of(bin_of(n ? n : 1));
}

if_pool_cfg if_pool_cfg_default(void) {
    if_pool_cfg c = {0, (size_t)64 << 20, 16384};
    return c;
}

/* ---- span 与池 ---- */
typedef struct span {
    char *base;
    size_t cap, bytes;
    uint64_t mask; /* slab：第 i 位 = 1 表示槽 i 空闲；大块：第 0 位 = 1 表示已缓存（空闲） */
    unsigned bin, slab;
    if_node bin_link; /* 挂在 bin[bin]：仍有空闲的 span */
    if_node all_link; /* 挂在 all（在用）或 spare（描述符回收） */
} span;

typedef struct pm_node {
    void *slot[PM_MASK + 1];
} pm_node;

struct if_pool {
    pthread_mutex_t lock;
    if_upstream up;
    if_pool_cfg cfg;
    arena meta;
    pm_node *top;
    if_node bin[BINS];
    if_node all, spare;
    if_pool_stats st;
    size_t self_bytes;
};

static int span_idle(const span *s) { return s->slab ? s->mask == FULL : (s->mask & 1) != 0; }

/* ---- 页表：地址 >> 12 -> span，三级各 12 位 ---- */
static span *pm_get(const if_pool *p, uintptr_t addr) {
    uint64_t page = addr >> PAGE_SHIFT;
    if (page >= PM_PAGES)
        return NULL;
    const pm_node *n = p->top;
    for (int level = 2; level > 0; --level) {
        n = n->slot[(page >> (level * PM_BITS)) & PM_MASK];
        if (!n)
            return NULL;
    }
    return n->slot[page & PM_MASK];
}

/* s 为 NULL 表示清除。 */
static int pm_set(if_pool *p, uint64_t page, span *s) {
    if (page >= PM_PAGES)
        return -EINVAL;
    pm_node *n = p->top;
    for (int level = 2; level > 0; --level) {
        void **slot = &n->slot[(page >> (level * PM_BITS)) & PM_MASK];
        if (!*slot) {
            if (!s)
                return 0;
            void *mem = NULL;
            int rc = arena_alloc(&p->meta, sizeof(pm_node), &mem);
            if (rc)
                return rc;
            *slot = mem;
        }
        n = *slot;
    }
    n->slot[page & PM_MASK] = s;
    return 0;
}

static int pm_set_range(if_pool *p, uintptr_t base, size_t bytes, span *s) {
    uint64_t first = base >> PAGE_SHIFT, n = bytes >> PAGE_SHIFT;
    for (uint64_t i = 0; i < n; ++i) {
        int rc = pm_set(p, first + i, s);
        if (rc) {
            for (uint64_t j = 0; j < i; ++j)
                pm_set(p, first + j, NULL);
            return rc;
        }
    }
    return 0;
}

/* ---- span 描述符 ---- */
static int meta_new(if_pool *p, span **out) {
    span *s;
    if (!if_list_empty(&p->spare)) {
        if_node *n = p->spare.next;
        if_list_del(n);
        s = IF_CONTAINER_OF(n, span, all_link);
    } else {
        void *mem = NULL;
        int rc = arena_alloc(&p->meta, sizeof(span), &mem);
        if (rc)
            return rc;
        s = mem;
    }
    memset(s, 0, sizeof *s);
    if_list_init(&s->bin_link);
    if_list_init(&s->all_link);
    *out = s;
    return 0;
}

/* 归还一个完全空闲的 span：先还上游，成功后才改元数据；失败时 span 原样保留。 */
static int span_release(if_pool *p, span *s) {
    int rc = p->up.unmap(p->up.ctx, s->base, s->bytes);
    if (rc)
        return rc;
    if (s->slab)
        pm_set_range(p, (uintptr_t)s->base, s->bytes, NULL);
    else
        pm_set(p, (uintptr_t)s->base >> PAGE_SHIFT, NULL);
    if_list_del(&s->bin_link);
    if_list_del(&s->all_link);
    if_list_add(&p->spare, &s->all_link);
    p->st.reserved_bytes -= s->bytes;
    p->st.idle_bytes -= s->bytes;
    ++p->st.unmaps;
    return 0;
}

static int trim_locked(if_pool *p) {
    int first = 0;
    for (if_node *n = p->all.next, *next; n != &p->all; n = next) {
        next = n->next;
        span *s = IF_CONTAINER_OF(n, span, all_link);
        if (!span_idle(s))
            continue;
        int rc = span_release(p, s);
        if (rc && !first)
            first = rc;
    }
    return first;
}

static int reserve_room(if_pool *p, size_t bytes) {
    size_t limit = p->cfg.max_reserved;
    if (!limit)
        return 0;
    if (bytes > limit)
        return -ENOMEM;
    if (p->st.reserved_bytes > limit - bytes)
        trim_locked(p);
    return p->st.reserved_bytes > limit - bytes ? -ENOMEM : 0;
}

/* 新建一个 span，挂进桶链，按「完全空闲」计入 idle_bytes。 */
static int span_new(if_pool *p, unsigned bin, span **out) {
    size_t cap = cap_of(bin);
    unsigned slab = cap <= p->cfg.small_max;
    size_t bytes = slab ? cap * SLOTS : IF_ALIGN_UP(cap, PAGE);
    int rc = reserve_room(p, bytes);
    if (rc)
        return rc;
    span *s;
    rc = meta_new(p, &s);
    if (rc)
        return rc;
    void *mem = NULL;
    rc = p->up.map(p->up.ctx, bytes, &mem);
    if (rc == -ENOMEM && p->st.idle_bytes) {
        trim_locked(p);
        rc = p->up.map(p->up.ctx, bytes, &mem);
    }
    if (!rc && ((uintptr_t)mem & (PAGE - 1))) {
        p->up.unmap(p->up.ctx, mem, bytes);
        rc = -EINVAL;
    }
    if (!rc) {
        s->base = mem;
        s->cap = cap;
        s->bytes = bytes;
        s->bin = bin;
        s->slab = slab;
        s->mask = slab ? FULL : 1;
        rc = slab ? pm_set_range(p, (uintptr_t)mem, bytes, s)
                  : pm_set(p, (uintptr_t)mem >> PAGE_SHIFT, s);
        if (rc)
            p->up.unmap(p->up.ctx, mem, bytes);
    }
    if (rc) {
        if_list_add(&p->spare, &s->all_link);
        return rc;
    }
    if_list_add(&p->bin[bin], &s->bin_link);
    if_list_add(&p->all, &s->all_link);
    p->st.reserved_bytes += bytes;
    p->st.idle_bytes += bytes;
    ++p->st.maps;
    *out = s;
    return 0;
}

/* 从有空闲的 span 里取一块；调用者保证 span 在桶链里。 */
static void *span_take(if_pool *p, span *s) {
    if (s->slab) {
        if (s->mask == FULL)
            p->st.idle_bytes -= s->bytes;
        unsigned slot = (unsigned)__builtin_ctzll(s->mask);
        s->mask &= s->mask - 1;
        if (!s->mask)
            if_list_del(&s->bin_link);
        return s->base + (size_t)slot * s->cap;
    }
    p->st.idle_bytes -= s->bytes;
    s->mask = 0;
    if_list_del(&s->bin_link);
    return s->base;
}

static int alloc_bin(if_pool *p, unsigned bin, void **out) {
    span *s;
    if (!if_list_empty(&p->bin[bin])) {
        s = IF_CONTAINER_OF(p->bin[bin].next, span, bin_link);
        ++p->st.reuses;
    } else {
        int rc = span_new(p, bin, &s);
        if (rc)
            return rc;
    }
    *out = span_take(p, s);
    p->st.active_bytes += s->cap;
    ++p->st.active_blocks;
    return 0;
}

static int free_span(if_pool *p, span *s, void *ptr) {
    size_t off = (size_t)((char *)ptr - s->base);
    if (s->slab) {
        if (off % s->cap)
            return -EINVAL;
        uint64_t bit = (uint64_t)1 << (off / s->cap);
        if (s->mask & bit)
            return -EINVAL; /* 重复释放 */
        if (!s->mask)
            if_list_add(&p->bin[s->bin], &s->bin_link);
        s->mask |= bit;
    } else {
        if (off || (s->mask & 1))
            return -EINVAL;
        s->mask = 1;
        if_list_add(&p->bin[s->bin], &s->bin_link);
    }
    p->st.active_bytes -= s->cap;
    --p->st.active_blocks;
    if (!span_idle(s))
        return 0;
    p->st.idle_bytes += s->bytes;
    /* 超出缓存上限就还给上游；失败时 span 留在池里仍算空闲，等下次 trim。 */
    return p->st.idle_bytes > p->cfg.max_idle ? span_release(p, s) : 0;
}

static int free_locked(if_pool *p, void *ptr) {
    span *s = pm_get(p, (uintptr_t)ptr);
    return s ? free_span(p, s, ptr) : -EINVAL;
}

/* ---- 对外接口 ---- */
static void pool_free_memory(if_pool *p) {
    size_t self = p->self_bytes;
    arena_chunk *c = p->meta.chunks;
    pthread_mutex_destroy(&p->lock);
    while (c) {
        arena_chunk *next = c->next;
        munmap(c, c->bytes);
        c = next;
    }
    munmap(p, self);
}

int if_pool_create(const if_upstream *up, const if_pool_cfg *cfg, if_pool **out) {
    if (!out)
        return -EINVAL;
    *out = NULL;
    if (!up)
        up = &host_upstream;
    if (!up->map || !up->unmap)
        return -EINVAL;
    if_pool_cfg c = cfg ? *cfg : if_pool_cfg_default();
    if (c.small_max > SMALL_LIMIT)
        return -EINVAL;
    size_t bytes = IF_ALIGN_UP(sizeof(if_pool), PAGE);
    void *mem = NULL;
    int rc = host_map(NULL, bytes, &mem);
    if (rc)
        return rc;
    if_pool *p = mem; /* mmap 的内存已清零 */
    rc = pthread_mutex_init(&p->lock, NULL);
    if (rc) {
        munmap(p, bytes);
        return -rc;
    }
    p->up = *up;
    p->cfg = c;
    p->self_bytes = bytes;
    for (unsigned i = 0; i < BINS; ++i)
        if_list_init(&p->bin[i]);
    if_list_init(&p->all);
    if_list_init(&p->spare);
    void *top = NULL;
    rc = arena_alloc(&p->meta, sizeof(pm_node), &top);
    if (rc) {
        pool_free_memory(p);
        return rc;
    }
    p->top = top;
    *out = p;
    return 0;
}

int if_pool_destroy(if_pool *p) {
    if (!p)
        return -EINVAL;
    pthread_mutex_lock(&p->lock);
    int rc = p->st.active_blocks ? -EBUSY : trim_locked(p);
    pthread_mutex_unlock(&p->lock);
    if (rc)
        return rc;
    pool_free_memory(p);
    return 0;
}

int if_pool_alloc(if_pool *p, size_t n, void **out) {
    if (!p || !out)
        return -EINVAL;
    *out = NULL;
    if (n > MAX_BYTES)
        return -ENOMEM;
    pthread_mutex_lock(&p->lock);
    int rc = alloc_bin(p, bin_of(n ? n : 1), out);
    pthread_mutex_unlock(&p->lock);
    return rc;
}

int if_pool_alloc_aligned(if_pool *p, size_t align, size_t n, void **out) {
    if (!p || !out || !IF_IS_POW2(align) || align > PAGE)
        return -EINVAL;
    *out = NULL;
    if (n > MAX_BYTES)
        return -ENOMEM;
    /* 块地址 = 页对齐的 span 基址 + 槽号 * 容量，所以容量是 align 的倍数就够了。 */
    unsigned bin = bin_of(n ? n : 1);
    while (bin < BINS && cap_of(bin) % align)
        ++bin;
    if (bin >= BINS)
        return -ENOMEM;
    pthread_mutex_lock(&p->lock);
    int rc = alloc_bin(p, bin, out);
    pthread_mutex_unlock(&p->lock);
    return rc;
}

int if_pool_free(if_pool *p, void *ptr) {
    if (!ptr)
        return 0;
    if (!p)
        return -EINVAL;
    pthread_mutex_lock(&p->lock);
    int rc = free_locked(p, ptr);
    pthread_mutex_unlock(&p->lock);
    return rc;
}

int if_pool_try_free(if_pool *p, void *ptr) {
    if (!ptr)
        return 0;
    if (!p)
        return -EINVAL;
    pthread_mutex_lock(&p->lock);
    span *s = pm_get(p, (uintptr_t)ptr);
    int rc = s ? free_span(p, s, ptr) : 1;
    pthread_mutex_unlock(&p->lock);
    return rc;
}

void if_pool_freeze(if_pool *p) {
    if (p)
        pthread_mutex_lock(&p->lock);
}

void if_pool_thaw(if_pool *p) {
    if (p)
        pthread_mutex_unlock(&p->lock);
}

ssize_t if_pool_usable(if_pool *p, const void *ptr) {
    if (!p || !ptr)
        return -EINVAL;
    ssize_t r = -EINVAL;
    pthread_mutex_lock(&p->lock);
    const span *s = pm_get(p, (uintptr_t)ptr);
    if (s) {
        size_t off = (size_t)((const char *)ptr - s->base);
        int live = s->slab ? ((off % s->cap) == 0 && !(s->mask & ((uint64_t)1 << (off / s->cap))))
                           : (off == 0 && !(s->mask & 1));
        if (live)
            r = (ssize_t)s->cap;
    }
    pthread_mutex_unlock(&p->lock);
    return r;
}

int if_pool_trim(if_pool *p) {
    if (!p)
        return -EINVAL;
    pthread_mutex_lock(&p->lock);
    int rc = trim_locked(p);
    pthread_mutex_unlock(&p->lock);
    return rc;
}

void if_pool_get_stats(if_pool *p, if_pool_stats *out) {
    if (!p || !out)
        return;
    pthread_mutex_lock(&p->lock);
    *out = p->st;
    pthread_mutex_unlock(&p->lock);
}
