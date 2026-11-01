/* LD_PRELOAD 垫片：进程里所有 malloc / free 走一个 if_pool。只依赖 mem 层。
 *
 *   LD_PRELOAD=libinfer_preload.so ./any_program
 *
 * 做法：
 *   - 池在第一次分配时才建：构造函数之前就可能有人 malloc。建池只用 mmap 和互斥锁，不会递归进 malloc。
 *   - free / realloc / malloc_usable_size 先问池「这是不是你的块」。是就处理，不是就交给 glibc 的 __libc_*：
 *     垫片装入之前的分配、池建不起来时的分配、对齐超过 4096 的分配，都归 glibc。不用 dlsym 取原函数，
 *     因为 dlsym 自己可能分配；只有 malloc_usable_size 没有 __libc_ 版本，才在第一次需要时用 dlsym。
 *   - 池里的块都是 16 字节对齐。池会复用块，所以 calloc 一律清零。
 *   - fork：prepare 里锁住池，父子进程各自解锁，子进程不会遇到别的线程正持有的池锁。处理函数在加载时注册，
 *     通常是最早注册的，所以 prepare 阶段排在最后，其他处理函数里的分配不会撞上已经锁住的池。
 *   - 池里的块被 free 两次、或指针指到块中间：直接 abort，和 glibc 一样。
 * 取舍（没测过就不声称更快）：
 *   - 全进程一把池锁。线程多、小分配多时，glibc 的线程缓存会更快；垫片的价值是预算、统计、可换上游。
 *   - realloc 换档要拷贝；glibc 对大块用 mremap，这里没有。
 *   - 空闲块最多缓存 max_idle（默认 64 MiB），超出才还给系统。
 */
#define _GNU_SOURCE
#include "infer/mem.h"
#include <dlfcn.h>
#include <errno.h>
#include <malloc.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PAGE 4096u
#define DIE(msg) do { static const char m_[] = msg; ssize_t r_ = write(2, m_, sizeof m_ - 1); (void)r_; abort(); } while (0)

extern void *__libc_malloc(size_t n);
extern void *__libc_calloc(size_t a, size_t b);
extern void *__libc_realloc(void *ptr, size_t n);
extern void *__libc_memalign(size_t align, size_t n);
extern void __libc_free(void *ptr);

static if_pool *_Atomic g_pool;
static pthread_mutex_t g_init_lock = PTHREAD_MUTEX_INITIALIZER;

static if_pool *pool_slow(void) {
    pthread_mutex_lock(&g_init_lock);
    if_pool *p = atomic_load_explicit(&g_pool, memory_order_relaxed);
    if (!p && if_pool_create(NULL, NULL, &p) == 0)
        atomic_store_explicit(&g_pool, p, memory_order_release);
    pthread_mutex_unlock(&g_init_lock);
    return p;
}

/* 取池，没有就建；建不起来返回 NULL，调用者退回 glibc。 */
static inline if_pool *pool(void) {
    if_pool *p = atomic_load_explicit(&g_pool, memory_order_acquire);
    return IF_LIKELY(p) ? p : pool_slow();
}

/* 只看不建：池还不存在时，任何指针都不可能是池里的。 */
static inline if_pool *pool_peek(void) { return atomic_load_explicit(&g_pool, memory_order_acquire); }

static int pow2(size_t a) { return a && !(a & (a - 1)); }

static void *pool_alloc(if_pool *p, size_t n) {
    void *out = NULL;
    if (IF_UNLIKELY(if_pool_alloc(p, n, &out))) {
        errno = ENOMEM;
        return NULL;
    }
    return out;
}

/* 池支持到 4096 的对齐；更大的交给 glibc。不超过 16 的对齐，池里的块天然满足。 */
static void *alloc_aligned(size_t align, size_t n) {
    if_pool *p = pool();
    if (IF_UNLIKELY(!p) || align > PAGE)
        return __libc_memalign(align, n);
    if (align <= 16)
        return pool_alloc(p, n);
    void *out = NULL;
    if (if_pool_alloc_aligned(p, align, n, &out)) {
        errno = ENOMEM;
        return NULL;
    }
    return out;
}

void *malloc(size_t n) {
    if_pool *p = pool();
    return IF_LIKELY(p) ? pool_alloc(p, n) : __libc_malloc(n);
}

void *calloc(size_t a, size_t b) {
    size_t n;
    if_pool *p = pool();
    if (IF_UNLIKELY(!p))
        return __libc_calloc(a, b);
    if (IF_UNLIKELY(IF_MUL_OVF(a, b, &n))) {
        errno = ENOMEM;
        return NULL;
    }
    void *r = pool_alloc(p, n);
    if (r)
        memset(r, 0, n);
    return r;
}

void free(void *ptr) {
    if (!ptr)
        return;
    int saved = errno; /* free 不改 errno */
    if_pool *p = pool_peek();
    int rc = p ? if_pool_try_free(p, ptr) : 1;
    if (rc == 1)
        __libc_free(ptr);
    else if (rc == -EINVAL)
        DIE("infer-preload: free(): invalid pointer or double free\n");
    errno = saved; /* 其他负值是还上游失败，逻辑释放已生效，块留在池里 */
}

void *realloc(void *ptr, size_t n) {
    if (!ptr)
        return malloc(n);
    if_pool *p = pool_peek();
    ssize_t cap = p ? if_pool_usable(p, ptr) : -EINVAL;
    if (cap < 0)
        return __libc_realloc(ptr, n); /* 不是池里的块：归 glibc */
    if (!n) {
        free(ptr);
        return NULL;
    }
    if (if_pool_class_size(n) == (size_t)cap)
        return ptr; /* 同一档，原地 */
    void *q = pool_alloc(p, n);
    if (!q)
        return NULL; /* 失败时原块不动 */
    memcpy(q, ptr, n < (size_t)cap ? n : (size_t)cap);
    free(ptr);
    return q;
}

void *reallocarray(void *ptr, size_t a, size_t b) {
    size_t n;
    if (IF_MUL_OVF(a, b, &n)) {
        errno = ENOMEM;
        return NULL;
    }
    return realloc(ptr, n);
}

/* memalign 对不是 2 的幂的对齐，glibc 会向上取整；这种情况原样交给它。 */
void *memalign(size_t align, size_t n) {
    return pow2(align) ? alloc_aligned(align, n) : __libc_memalign(align, n);
}

void *aligned_alloc(size_t align, size_t n) {
    if (!pow2(align)) {
        errno = EINVAL;
        return NULL;
    }
    return alloc_aligned(align, n);
}

int posix_memalign(void **out, size_t align, size_t n) {
    if (!pow2(align) || align % sizeof(void *))
        return EINVAL;
    int saved = errno; /* posix_memalign 通过返回值报错，不改 errno */
    void *r = alloc_aligned(align, n);
    errno = saved;
    if (!r)
        return ENOMEM;
    *out = r;
    return 0;
}

void *valloc(size_t n) { return alloc_aligned(PAGE, n); }

void *pvalloc(size_t n) {
    if (n > SIZE_MAX - PAGE) {
        errno = ENOMEM;
        return NULL;
    }
    return alloc_aligned(PAGE, IF_ALIGN_UP(n, (size_t)PAGE));
}

/* RTLD_NEXT resolution is outside the pool lock. TLS avoids recursive
 * pthread_once if the loader itself asks about a foreign allocation. */
static pthread_once_t usable_once = PTHREAD_ONCE_INIT;
static size_t (*usable_fn)(void *);
static _Thread_local int resolving_usable;
static void resolve_usable(void) {
    usable_fn = (size_t (*)(void *))dlsym(RTLD_NEXT, "malloc_usable_size");
}
static size_t libc_usable(void *ptr) {
    if (resolving_usable) return 0; /* no portable libc bootstrap symbol */
    int saved = errno;
    resolving_usable = 1;
    pthread_once(&usable_once, resolve_usable);
    resolving_usable = 0;
    size_t result = usable_fn ? usable_fn(ptr) : 0;
    errno = saved;
    return result;
}

size_t malloc_usable_size(void *ptr) {
    if (!ptr)
        return 0;
    if_pool *p = pool_peek();
    ssize_t cap = p ? if_pool_usable(p, ptr) : -EINVAL;
    return cap >= 0 ? (size_t)cap : libc_usable(ptr);
}

static void fork_prepare(void) { if_pool_freeze(pool_peek()); }
static void fork_release(void) { if_pool_thaw(pool_peek()); }

/* 先建池再注册：pthread_atfork 自己会分配，那次分配要落在已经就绪的池里。 */
static void __attribute__((constructor)) shim_load(void) {
    if (pool())
        pthread_atfork(fork_prepare, fork_release, fork_release);
}

/* 测试和诊断用：垫片在不在、池里现在有多少块。没有垫片时这个符号不存在。 */
IF_API int if_preload_stats(if_pool_stats *out) {
    if_pool *p = pool_peek();
    if (!p || !out)
        return -EINVAL;
    if_pool_get_stats(p, out);
    return 0;
}
