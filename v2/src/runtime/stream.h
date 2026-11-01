#pragma once
#include <memory>
namespace infer_c {
struct Stream {
    void *value = nullptr;
    int device;
    explicit Stream(int device);
    ~Stream();
};
}
