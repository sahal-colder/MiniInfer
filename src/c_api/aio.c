/* 第 4 层：store 之上的异步执行器。一把锁、一张固定大小的作业表、一个按代号先进先出的工作线程。 */
#include "infer/aio.h"
#include "lifetime.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

enum { JOB_FREE, JOB_QUEUED, JOB_RUNNING, JOB_DONE };

typedef struct job {
    uint64_t token;
    int state, op, rc;
    size_t n;
    if_store_req r[IF_AIO_MAX_REQ];
} job;

struct if_aio {
    pthread_mutex_t lock;
    pthread_cond_t wake; /* 工作线程等作业 */
    pthread_cond_t done; /* 调用方等完成 */
    pthread_t worker;
    if_store *store;
    int stopping;
    uint64_t next;
    if_aio_stats st;
    job jobs[IF_AIO_MAX_JOBS];
};

/* 只在作业处于 RUNNING 时由工作线程访问它的请求数组，不需要锁。 */
static void run_job(if_store *s, job *j) {
    int first = 0;
    switch (j->op) {
    case IF_AIO_WRITE:
        first = if_store_write_many(s, j->r, j->n);
        break;
    case IF_AIO_READ:
        first = if_store_read_many(s, j->r, j->n);
        break;
    default:
        for (size_t i = 0; i < j->n; ++i) {
            j->r[i].rc = if_store_prefetch(s, j->r[i].slot);
            if (j->r[i].rc < 0 && !first)
                first = j->r[i].rc;
        }
    }
    j->rc = first;
}

/* 排队中代号最小的作业；调用时持有锁。 */
static job *next_queued(if_aio *a) {
    job *best = NULL;
    for (size_t i = 0; i < IF_AIO_MAX_JOBS; ++i)
        if (a->jobs[i].state == JOB_QUEUED && (!best || a->jobs[i].token < best->token))
            best = &a->jobs[i];
    return best;
}

static void *worker_main(void *arg) {
    if_aio *a = arg;
    pthread_mutex_lock(&a->lock);
    for (;;) {
        job *j = next_queued(a);
        if (!j) {
            if (a->stopping)
                break;
            pthread_cond_wait(&a->wake, &a->lock);
            continue;
        }
        j->state = JOB_RUNNING;
        pthread_mutex_unlock(&a->lock);
        run_job(a->store, j);
        pthread_mutex_lock(&a->lock);
        j->state = JOB_DONE;
        ++a->st.completed;
        pthread_cond_broadcast(&a->done);
    }
    pthread_mutex_unlock(&a->lock);
    return NULL;
}

static job *find(if_aio *a, uint64_t token) {
    for (size_t i = 0; i < IF_AIO_MAX_JOBS; ++i)
        if (a->jobs[i].state != JOB_FREE && a->jobs[i].token == token)
            return &a->jobs[i];
    return NULL;
}

/* 作业已完成：取走结果并释放位置。 */
static int consume(if_aio *a, job *j, if_store_req *out) {
    int rc = j->rc;
    if (out)
        memcpy(out, j->r, j->n * sizeof *out);
    j->state = JOB_FREE;
    --a->st.inflight;
    return rc;
}

int if_aio_create(if_store *s, if_aio **out) {
    if (!out)
        return -EINVAL;
    *out = NULL;
    if (!s)
        return -EINVAL;
    if_aio *a = calloc(1, sizeof *a);
    if (!a)
        return -ENOMEM;
    a->store = s;
    a->next = 1;
    int rc = pthread_mutex_init(&a->lock, NULL);
    if (rc) { free(a); return -rc; }
    rc = pthread_cond_init(&a->wake, NULL);
    if (rc) { pthread_mutex_destroy(&a->lock); free(a); return -rc; }
    rc = pthread_cond_init(&a->done, NULL);
    if (rc) {
        pthread_cond_destroy(&a->wake);
        pthread_mutex_destroy(&a->lock);
        free(a);
        return -rc;
    }
    if_store_retain(s);
    rc = pthread_create(&a->worker, NULL, worker_main, a);
    if (rc) {
        if_store_unref(s);
        pthread_cond_destroy(&a->done);
        pthread_cond_destroy(&a->wake);
        pthread_mutex_destroy(&a->lock);
        free(a);
        return -rc;
    }
    *out = a;
    return 0;
}

/* 工作线程只在没有排队作业且 stopping 时退出，所以已接受的作业都会做完。 */
int if_aio_destroy(if_aio *a) {
    if (!a)
        return -EINVAL;
    pthread_mutex_lock(&a->lock);
    a->stopping = 1;
    pthread_cond_signal(&a->wake);
    pthread_mutex_unlock(&a->lock);
    pthread_join(a->worker, NULL);
    if_store_unref(a->store);
    pthread_cond_destroy(&a->done);
    pthread_cond_destroy(&a->wake);
    pthread_mutex_destroy(&a->lock);
    free(a);
    return 0;
}

int if_aio_submit(if_aio *a, int op, const if_store_req *r, size_t n, uint64_t *token) {
    if (!a || !r || !token || !n || n > IF_AIO_MAX_REQ || op < IF_AIO_WRITE || op > IF_AIO_PREFETCH)
        return -EINVAL;
    pthread_mutex_lock(&a->lock);
    job *j = NULL;
    for (size_t i = 0; i < IF_AIO_MAX_JOBS && !j; ++i)
        if (a->jobs[i].state == JOB_FREE)
            j = &a->jobs[i];
    if (!j) {
        ++a->st.rejected;
        pthread_mutex_unlock(&a->lock);
        return -EBUSY;
    }
    j->token = a->next++;
    j->op = op;
    j->n = n;
    j->rc = 0;
    memcpy(j->r, r, n * sizeof *r);
    j->state = JOB_QUEUED;
    ++a->st.submitted;
    if (++a->st.inflight > a->st.peak)
        a->st.peak = a->st.inflight;
    *token = j->token;
    pthread_cond_signal(&a->wake);
    pthread_mutex_unlock(&a->lock);
    return 0;
}

int if_aio_wait(if_aio *a, uint64_t token, if_store_req *out) {
    int rc = 0;
    if (!a)
        return -EINVAL;
    pthread_mutex_lock(&a->lock);
    for (;;) {
        job *j = find(a, token);
        if (!j) {
            rc = -ENOENT;
            break;
        }
        if (j->state == JOB_DONE) {
            rc = consume(a, j, out);
            break;
        }
        pthread_cond_wait(&a->done, &a->lock);
    }
    pthread_mutex_unlock(&a->lock);
    return rc;
}

int if_aio_poll(if_aio *a, uint64_t token, if_store_req *out) {
    int rc;
    if (!a)
        return -EINVAL;
    pthread_mutex_lock(&a->lock);
    job *j = find(a, token);
    if (!j)
        rc = -ENOENT;
    else if (j->state != JOB_DONE)
        rc = 1;
    else
        rc = consume(a, j, out);
    pthread_mutex_unlock(&a->lock);
    return rc;
}

int if_aio_drain(if_aio *a) {
    int first = 0;
    uint64_t first_token = UINT64_MAX;
    if (!a)
        return -EINVAL;
    pthread_mutex_lock(&a->lock);
    for (;;) {
        int busy = 0;
        for (size_t i = 0; i < IF_AIO_MAX_JOBS; ++i)
            busy |= a->jobs[i].state == JOB_QUEUED || a->jobs[i].state == JOB_RUNNING;
        if (!busy)
            break;
        pthread_cond_wait(&a->done, &a->lock);
    }
    for (size_t i = 0; i < IF_AIO_MAX_JOBS; ++i) {
        job *j = &a->jobs[i];
        if (j->state != JOB_DONE)
            continue;
        if (j->rc && j->token < first_token) {
            first = j->rc;
            first_token = j->token;
        }
        j->state = JOB_FREE;
        --a->st.inflight;
    }
    pthread_mutex_unlock(&a->lock);
    return first;
}

void if_aio_get_stats(if_aio *a, if_aio_stats *out) {
    if (!a || !out)
        return;
    pthread_mutex_lock(&a->lock);
    *out = a->st;
    pthread_mutex_unlock(&a->lock);
}
