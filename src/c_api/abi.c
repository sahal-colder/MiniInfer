/* 第 5 层：abi。各层头文件本身就是 ABI，这里只补版本号入口；libinfer.so 由 CMake 把各层合在一起。 */
#include "infer/infer.h"

int if_abi_version(void) { return IF_ABI_VERSION; }
