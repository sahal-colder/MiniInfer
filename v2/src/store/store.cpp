/* Fixed scheduler namespace facade over the shared logical storage core. */
#include "infer/store.h"
#include "infer/storage/kv_offload_cache.h"
#include "../runtime/boundary.h"
#include "lifetime.h"
#include <atomic>
#include <limits>

struct if_store {
    if_store_cfg cfg;
    std::shared_ptr<infer_c::HostMemory> memory;
    std::unique_ptr<infer::KVOffloadCache> cache;
    if_dev *owned = nullptr;
    std::atomic<size_t> users{0};
    std::atomic<uint64_t> verify_failures{0}, prefetches{0};
};
static size_t capacity(size_t bytes) {
    return infer::CPUStorageBackend::allocation_size(std::max(bytes,size_t(4096)));
}
uint64_t if_store_dev_bytes(size_t bytes,size_t slots) {
    if (!bytes || bytes>(size_t(1)<<40) || !slots) return 0;
    size_t total=0;
    size_t record=capacity(bytes)+infer::SSDStorageBackend::record_header_bytes;
    return IF_MUL_OVF(record,slots,&total) || total>size_t(INT64_MAX) ? 0 : total;
}
int if_store_create(const if_store_cfg *cfg,if_store **out) {
    if (!out) return -EINVAL;
    *out=nullptr;
    if (!cfg || !cfg->block_bytes || cfg->block_bytes>(size_t(1)<<40) || !cfg->slots || !cfg->cpu_slots ||
        cfg->cpu_slots>cfg->slots || (cfg->ssd && cfg->ssd_dir)) return -EINVAL;
    uint64_t need=if_store_dev_bytes(cfg->block_bytes,cfg->slots);
    if ((cfg->ssd || cfg->ssd_dir) && !need) return -EOVERFLOW;
    if (cfg->ssd && cfg->ssd->size<need) return -ENOSPC;
    return infer_c::status([&] {
        auto s=std::make_unique<if_store>();
        s->cfg=*cfg;
        s->memory=std::make_shared<infer_c::HostMemory>(cfg->cpu_up);
        if_dev *dev=cfg->ssd;
        if (cfg->ssd_dir) {
            infer_c::require(if_dev_open_tmp(cfg->ssd_dir,need,&s->owned));
            dev=s->owned;
        }
        infer::KVOffloadConfig c;
        c.payload_bytes=cfg->block_bytes;
        c.logical_slots=cfg->slots;
        c.cpu_slots=cfg->cpu_slots;
        c.compact_slots=true;
        c.variable_payloads=true;
        c.enable_ssd=dev!=nullptr;
        c.promote_hits=cfg->promote_hits;
        c.promote_evict=true;
        c.write_concurrency=1;
        c.retain_ssd_copy=true;
        c.ssd_verification=infer::SSDVerification::ON_READ;
        /* Device creation is the only owned platform resource before cache.
         * Close explicitly on failure so no failed constructor leaks the fd. */
        int rc=infer_c::status([&] {
            s->cache=std::make_unique<infer::KVOffloadCache>(c,infer_c::cpu(capacity(cfg->block_bytes),s->memory),
                infer_c::disk(capacity(cfg->block_bytes),need,dev,cfg->piece_bytes ? cfg->piece_bytes : 128*1024));
            return 0;
        });
        if (rc) {
            if_dev_close(s->owned);
            return rc;
        }
        *out=s.release();
        return 0;
    });
}
int if_store_destroy(if_store *s) {
    if (!s) return -EINVAL;
    if (s->users.load()) return -EBUSY;
    s->cache.reset();
    int rc=s->memory->pool ? s->memory->close() : 0;
    if (rc) return rc; // retained object can retry physical cleanup
    rc=if_dev_close(s->owned);
    delete s;
    return rc;
}
void if_store_retain(if_store *s) { s->users.fetch_add(1); }
void if_store_unref(if_store *s) { s->users.fetch_sub(1); }
int if_store_write(if_store *s,size_t slot,const void *buf,size_t bytes) {
    if (!s || !buf || !bytes) return -EINVAL;
    if (slot>=s->cfg.slots || bytes>s->cfg.block_bytes) return -ERANGE;
    return infer_c::status([&] {
        s->cache->write_many({slot},{const_cast<void *>(buf)},{bytes});
        return 0;
    });
}
ssize_t if_store_read(if_store *s,size_t slot,void *buf,size_t cap) {
    if (!s || !buf) return -EINVAL;
    if (slot>=s->cfg.slots) return -ERANGE;
    size_t bytes=0;
    int rc=infer_c::status([&] {
        if (!s->cache->metadata(slot).id) return -ENOENT;
        bytes=s->cache->read(slot,buf,cap);
        return 0;
    });
    if (rc==-EBADMSG) ++s->verify_failures;
    return rc ? rc : static_cast<ssize_t>(bytes);
}
int if_store_write_many(if_store *s,if_store_req *r,size_t n) {
    if (!s || (n && !r)) return -EINVAL;
    int first=0;
    for (size_t i=0;i<n;++i) {
        r[i].rc=if_store_write(s,r[i].slot,r[i].buf,r[i].len);
        if (!first && r[i].rc) first=r[i].rc;
    }
    return first;
}
int if_store_read_many(if_store *s,if_store_req *r,size_t n) {
    if (!s || (n && !r)) return -EINVAL;
    int first=0;
    for (size_t i=0;i<n;++i) {
        ssize_t rc=if_store_read(s,r[i].slot,r[i].buf,r[i].len);
        r[i].rc=rc<0 ? static_cast<int>(rc) : 0;
        if (rc>=0) r[i].len=static_cast<size_t>(rc);
        if (!first && r[i].rc) first=r[i].rc;
    }
    return first;
}
int if_store_erase(if_store *s,size_t slot) {
    if (!s) return -EINVAL;
    if (slot>=s->cfg.slots) return -ERANGE;
    return infer_c::status([&] { return s->cache->erase(slot) ? 0 : -ENOENT; });
}
int if_store_prefetch(if_store *s,size_t slot) {
    if (!s) return -EINVAL;
    if (slot>=s->cfg.slots) return -ERANGE;
    int rc=infer_c::status([&] {
        auto meta=s->cache->metadata(slot);
        if (!meta.id) return -ENOENT;
        bool ready=s->cache->prefetch(slot,false);
        if (ready && meta.location==infer::StorageLocation::SSD) ++s->prefetches;
        return ready ? 1 : 0;
    });
    if (rc==-EBADMSG) ++s->verify_failures;
    return rc;
}
int if_store_where(if_store *s,size_t slot) {
    if (!s) return -EINVAL;
    if (slot>=s->cfg.slots) return -ERANGE;
    return infer_c::status([&] {
        auto m=s->cache->metadata(slot);
        if (!m.id) return 0;
        return (m.location==infer::StorageLocation::CPU ? IF_AT_CPU : IF_AT_SSD) | (m.ssd_copy ? IF_AT_SSD : 0);
    });
}
void if_store_get_stats(if_store *s,if_store_stats *out) {
    if (!s || !out) return;
    /* Valid handles and internal snapshots cannot fail under normal operation;
     * never allow an allocator/statistics exception to cross the C boundary. */
    int rc=infer_c::status([&] {
        auto st=s->cache->stats();
        *out={};
        out->writes=st.writes; out->cpu_hits=st.cpu_hits; out->ssd_hits=st.ssd_hits;
        out->demotions=st.ssd_record_writes;
        out->evictions=st.demotions>out->demotions ? st.demotions-out->demotions : 0;
        out->promotions=st.promotions;
        out->prefetches=s->prefetches.load(); out->verify_failures=s->verify_failures.load();
        out->ssd_written_bytes=st.ssd_payload_written_bytes+64*st.ssd_record_writes;
        out->ssd_read_bytes=st.ssd_payload_read_bytes;
        out->populated=st.populated_slots;
        out->cpu_blocks=st.cpu_reserved_bytes/capacity(s->cfg.block_bytes);
        if_pool_get_stats(s->memory->pool,&out->pool);
        return 0;
    });
    if (rc) *out={};
}
