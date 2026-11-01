/* Stable logical Block API. No platform headers or physical pool handles.
 * Buffers are borrowed until synchronous completion. -errno on failure.
 * destroy must not race calls on the same outer handle. Leases and accepted
 * migrations retain core state after runtime destruction. */
#ifndef INFER_RUNTIME_H
#define INFER_RUNTIME_H
#include "infer/io.h"
#include "infer/mem.h"
#ifdef __cplusplus
extern "C" {
#endif

typedef uint64_t if_block_id;
typedef struct if_runtime if_runtime;
typedef struct if_lease if_lease;
typedef struct if_migration if_migration;
enum { IF_TARGET_CPU = 0, IF_TARGET_GPU = 1, IF_TARGET_SSD = 2 };
enum { IF_ACCESS_READ = 0, IF_ACCESS_WRITE = 1 };

typedef struct if_runtime_cfg {
    size_t block_bytes;
    size_t cpu_bytes, gpu_bytes, ssd_bytes;
    const char *ssd_dir;
    const if_upstream *cpu_up; /* copied callbacks; ctx must outlive runtime and leases */
    if_dev *ssd;              /* borrowed; outlives runtime, late leases and accepted work */
    int gpu_device;
} if_runtime_cfg;
typedef struct if_migration_result {
    int committed, cleanup_pending, error;
} if_migration_result;
typedef struct if_block_meta {
    size_t size, leases;
    uint64_t version;
    int target, state;
} if_block_meta;

IF_API int if_runtime_create(const if_runtime_cfg *cfg, if_runtime **out);
/* Provider teardown errors retain the outer handle; only retry destroy.
 * With late leases, teardown is deferred to their checked finish calls. */
IF_API int if_runtime_destroy(if_runtime *r);
IF_API int if_runtime_allocate(if_runtime *r, size_t bytes, int target, if_block_id *out);
IF_API int if_runtime_release(if_runtime *r, if_block_id id);
/* target access migrates as needed. Existing leases return -EBUSY rather than
 * moving their backing. GPU leases use one backend-owned consumer stream;
 * if_lease_stream returns that opaque CUDA stream for compute submission. */
IF_API int if_runtime_acquire(if_runtime *r, if_block_id id, int target, int mode, if_lease **out);
IF_API const void *if_lease_data(const if_lease *lease);
IF_API void *if_lease_writable_data(if_lease *lease);
IF_API size_t if_lease_size(const if_lease *lease);
IF_API void *if_lease_stream(const if_lease *lease);
/* Success consumes access, not the handle; destroy the handle afterwards.
 * Completion failure retains access so finish may be retried. Final provider
 * cleanup failure retains the handle after access ends; retry finish.
 * Destroy abandons WRITE and performs best-effort cleanup; use finish to
 * observe and retry errors before destroying the handle. */
IF_API int if_lease_finish(if_lease *lease);
IF_API void if_lease_destroy(if_lease *lease);
IF_API ssize_t if_runtime_read(if_runtime *r, if_block_id id, void *buf, size_t cap);
IF_API int if_runtime_write(if_runtime *r, if_block_id id, const void *buf, size_t bytes);
IF_API int if_runtime_metadata(if_runtime *r, if_block_id id, if_block_meta *out);
IF_API int if_runtime_migrate(if_runtime *r, if_block_id id, int target, if_migration_result *out);
IF_API int if_runtime_migrate_async(if_runtime *r, if_block_id id, int target, if_migration **out);
/* A completed result can be queried repeatedly until destroy. */
IF_API int if_migration_poll(if_migration *m, if_migration_result *out);
IF_API int if_migration_wait(if_migration *m, if_migration_result *out);
IF_API void if_migration_destroy(if_migration *m);
IF_API int if_runtime_retry_cleanup(if_runtime *r, if_block_id id);
#ifdef __cplusplus
}
#endif
#endif
