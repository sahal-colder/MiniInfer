#ifndef INFER_STORE_LIFETIME_H
#define INFER_STORE_LIFETIME_H
#include "infer/store.h"
#ifdef __cplusplus
extern "C" {
#endif
void if_store_retain(if_store *s);
void if_store_unref(if_store *s);
#ifdef __cplusplus
}
#endif
#endif
