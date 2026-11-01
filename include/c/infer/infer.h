/* infer/infer.h -- 总头文件，也是共享库 libinfer.so 的 ABI 面。
 *
 * 每一层的头文件本身就是 C ABI：只有基本整数、指针和 POD 结构，成功返回 >= 0，失败返回 -errno，没有异常，
 * 不依赖全局 errno。绑定层（Python ctypes、C++ 适配器）只包含这个头，或者 dlopen 之后按名字取符号。
 *   物理内存和 DeviceIO 独立；runtime / store 复用同一个 C++ Storage Core，aio 调用 store。
 * 规则：
 *   - 导出符号只有头文件里标了 IF_API 的函数，全部以 if_ 开头；其余符号在库内隐藏。
 *   - 任何结构体或函数的改动都让 IF_ABI_VERSION 加一。调用方启动时核对 if_abi_version() == IF_ABI_VERSION，
 *     不一致就不要使用这个库：结构体由调用方按头文件的布局传入，版本不符会读写到错的偏移。
 */
#ifndef INFER_H
#define INFER_H

#include "infer/base.h"
#include "infer/mem.h"
#include "infer/io.h"
#include "infer/store.h"
#include "infer/aio.h"
#include "infer/runtime.h"

#define IF_ABI_VERSION 3

#ifdef __cplusplus
extern "C" {
#endif

IF_API int if_abi_version(void);

#ifdef __cplusplus
}
#endif
#endif
