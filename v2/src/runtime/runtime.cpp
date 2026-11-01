#include "infer/runtime.h"
#include "boundary.h"
#include "stream.h"
#include <chrono>
#include <cstring>
#include <mutex>

struct if_runtime {
    std::shared_ptr<infer_c::HostMemory> memory;
    std::shared_ptr<infer_c::Stream> stream;
    std::unique_ptr<infer::TieredStorage> core;
};
struct if_lease {
    std::shared_ptr<infer_c::HostMemory> memory;
    std::shared_ptr<infer_c::Stream> stream;
    infer::BlockLease access; // destroyed before stream
    bool writable;
};
struct if_migration { std::shared_future<infer::MigrationResult> result; };

static infer::StorageLocation target(int value) {
    if (value == IF_TARGET_CPU) return infer::StorageLocation::CPU;
    if (value == IF_TARGET_GPU) return infer::StorageLocation::GPU;
    if (value == IF_TARGET_SSD) return infer::StorageLocation::SSD;
    throw std::invalid_argument("Unknown access target");
}
static int migration_result(const infer::MigrationResult& result, if_migration_result *out) {
    int rc = 0;
    if (result.error) {
        try { std::rethrow_exception(result.error); } catch (...) { rc = infer_c::error(); }
    }
    *out = {result.committed,result.cleanup_pending,rc};
    return rc;
}
int if_runtime_create(const if_runtime_cfg *cfg, if_runtime **out) {
    if (!out) return -EINVAL;
    *out = nullptr;
    if (!cfg || !cfg->block_bytes || (!cfg->cpu_bytes && !cfg->gpu_bytes) || (cfg->ssd && cfg->ssd_dir)) return -EINVAL;
    return infer_c::status([&] {
        infer::StorageConfig c;
        c.block_bytes=cfg->block_bytes;
        c.budget={cfg->cpu_bytes,cfg->gpu_bytes,cfg->ssd_bytes};
        c.gpu_device=cfg->gpu_device;
        c.ssd_directory=cfg->ssd_dir ? cfg->ssd_dir : "/tmp";
        c.ssd_verification=infer::SSDVerification::ON_READ;
        auto r=std::make_unique<if_runtime>();
        if (cfg->cpu_up) r->memory=std::make_shared<infer_c::HostMemory>(cfg->cpu_up);
        r->core=std::make_unique<infer::TieredStorage>(c,r->memory ? infer_c::cpu(c.block_bytes,r->memory) : nullptr,
            nullptr,infer_c::disk(c.block_bytes,c.budget.ssd_bytes,cfg->ssd));
        if (cfg->gpu_bytes) r->stream=std::make_shared<infer_c::Stream>(cfg->gpu_device);
        *out=r.release();
        return 0;
    });
}
int if_runtime_destroy(if_runtime *r) {
    if (!r) return -EINVAL;
    return infer_c::status([&] {
        r->core.reset();
        int rc=r->memory ? r->memory->retire() : 0;
        if (rc) return rc;
        delete r;
        return 0;
    });
}
int if_runtime_allocate(if_runtime *r,size_t bytes,int at,if_block_id *out) {
    if (!r || !out) return -EINVAL;
    *out=0;
    return infer_c::status([&] { *out=r->core->create(bytes,target(at)); return 0; });
}
int if_runtime_release(if_runtime *r,if_block_id id) {
    if (!r) return -EINVAL;
    return infer_c::status([&] { return r->core->erase(id) ? 0 : -EBUSY; });
}
int if_runtime_acquire(if_runtime *r,if_block_id id,int at,int mode,if_lease **out) {
    if (!out) return -EINVAL;
    *out=nullptr;
    if (!r || (mode!=IF_ACCESS_READ && mode!=IF_ACCESS_WRITE) || at==IF_TARGET_SSD) return -EINVAL;
    return infer_c::status([&] {
        auto to=target(at);
        if (r->core->metadata(id).location!=to) {
            if_migration_result result{};
            auto rc=migration_result(r->core->migrate(id,to),&result);
            if (rc) return rc;
            if (!result.committed || result.cleanup_pending) return -EBUSY;
        }
        auto lease=std::make_unique<if_lease>();
        lease->memory=r->memory;
        lease->stream=at==IF_TARGET_GPU ? r->stream : nullptr;
        lease->writable=mode==IF_ACCESS_WRITE;
        lease->access=r->core->acquire_at(id,to,lease->writable ? infer::AccessMode::WRITE : infer::AccessMode::READ,
            lease->stream ? static_cast<cudaStream_t>(lease->stream->value) : nullptr);
        *out=lease.release();
        return 0;
    });
}
const void *if_lease_data(const if_lease *l) { return l ? l->access.data() : nullptr; }
void *if_lease_writable_data(if_lease *l) {
    if (!l || !l->writable || !l->access) return nullptr;
    return l->access.writable_data();
}
size_t if_lease_size(const if_lease *l) { return l ? l->access.size() : 0; }
void *if_lease_stream(const if_lease *l) { return l && l->stream ? l->stream->value : nullptr; }
int if_lease_finish(if_lease *l) {
    if (!l) return -EINVAL;
    return infer_c::status([&] {
        l->access.finish();
        return l->memory ? l->memory->finish() : 0;
    });
}
void if_lease_destroy(if_lease *l) { delete l; }
ssize_t if_runtime_read(if_runtime *r,if_block_id id,void *buf,size_t cap) {
    if (!r || !buf) return -EINVAL;
    size_t bytes=0;
    int rc=infer_c::status([&] {
        auto meta=r->core->metadata(id);
        bytes=meta.size;
        if (cap<bytes) return -ENOBUFS;
        if (meta.location==infer::StorageLocation::GPU) {
            if_lease *l=nullptr;
            int status=if_runtime_acquire(r,id,IF_TARGET_CPU,IF_ACCESS_READ,&l);
            if (status) return status;
            std::unique_ptr<if_lease> lease(l);
            std::memcpy(buf,if_lease_data(l),bytes);
            return if_lease_finish(l);
        }
        r->core->read_to_host(id,buf,bytes);
        return 0;
    });
    return rc ? rc : static_cast<ssize_t>(bytes);
}
int if_runtime_write(if_runtime *r,if_block_id id,const void *buf,size_t bytes) {
    if (!r || !buf) return -EINVAL;
    return infer_c::status([&] {
        if (r->core->metadata(id).size!=bytes) return -EINVAL;
        if_lease *l=nullptr;
        int rc=if_runtime_acquire(r,id,IF_TARGET_CPU,IF_ACCESS_WRITE,&l);
        if (rc) return rc;
        std::unique_ptr<if_lease> lease(l);
        std::memcpy(if_lease_writable_data(l),buf,bytes);
        return if_lease_finish(l);
    });
}
int if_runtime_metadata(if_runtime *r,if_block_id id,if_block_meta *out) {
    if (!r || !out) return -EINVAL;
    return infer_c::status([&] {
        auto m=r->core->metadata(id);
        *out={m.size,m.lease_count,m.version,m.location==infer::StorageLocation::CPU ? IF_TARGET_CPU :
            m.location==infer::StorageLocation::GPU ? IF_TARGET_GPU : IF_TARGET_SSD,static_cast<int>(m.state)};
        return 0;
    });
}
int if_runtime_migrate(if_runtime *r,if_block_id id,int at,if_migration_result *out) {
    if (!r || !out) return -EINVAL;
    *out={};
    int rc=infer_c::status([&] { return migration_result(r->core->migrate(id,target(at)),out); });
    if (rc) out->error=rc;
    return rc;
}
int if_runtime_migrate_async(if_runtime *r,if_block_id id,int at,if_migration **out) {
    if (!out) return -EINVAL;
    *out=nullptr;
    if (!r) return -EINVAL;
    return infer_c::status([&] {
        auto m=std::make_unique<if_migration>();
        m->result=r->core->migrate_async(id,target(at)).share();
        *out=m.release();
        return 0;
    });
}
int if_migration_poll(if_migration *m,if_migration_result *out) {
    if (!m || !out) return -EINVAL;
    return infer_c::status([&] {
        if (m->result.wait_for(std::chrono::seconds(0))!=std::future_status::ready) return 1;
        return if_migration_wait(m,out);
    });
}
int if_migration_wait(if_migration *m,if_migration_result *out) {
    if (!m || !out) return -EINVAL;
    return infer_c::status([&] { return migration_result(m->result.get(),out); });
}
void if_migration_destroy(if_migration *m) { delete m; }
int if_runtime_retry_cleanup(if_runtime *r,if_block_id id) {
    if (!r) return -EINVAL;
    return infer_c::status([&] { return r->core->retry_cleanup(id) ? 0 : -EBUSY; });
}
