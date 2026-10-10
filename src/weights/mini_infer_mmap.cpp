#include "mini_infer/weights/mini_infer_mmap.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mini_infer {

MappedModelFile::MappedModelFile(const std::string& path) {
    // 第一步：以只读方式打开。O_CLOEXEC 避免子进程继承描述符。
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        throw std::system_error(errno, std::generic_category(),
                                "open model file: " + path);
    }

    // 第二步：用同一个 fd 查询大小。课件用 fopen/fseek/ftell 再开一次；
    // fstat 避免两次打开之间文件被替换，也避免忘记 fclose。
    struct stat file_stat {};
    if (::fstat(fd, &file_stat) != 0) {
        const int error = errno;
        ::close(fd);
        throw std::system_error(error, std::generic_category(),
                                "fstat model file: " + path);
    }
    if (!S_ISREG(file_stat.st_mode) ||
        file_stat.st_size < static_cast<off_t>(sizeof(ModelConfig)) ||
        static_cast<uintmax_t>(file_stat.st_size) >
            std::numeric_limits<std::size_t>::max()) {
        ::close(fd);
        throw std::runtime_error(
            "model file is not a regular file with a full header");
    }
    size_bytes_ = static_cast<std::size_t>(file_stat.st_size);

    // 第三步：映射整个文件。PROT_READ 禁止通过映射写入；MAP_PRIVATE
    // 表示私有映射。映射成功后，关闭 fd 不会使映射失效。
    mapping_ = ::mmap(nullptr, size_bytes_, PROT_READ, MAP_PRIVATE, fd, 0);
    const int map_error = errno;
    ::close(fd);
    if (mapping_ == MAP_FAILED) {
        mapping_ = nullptr;
        throw std::system_error(map_error, std::generic_category(),
                                "mmap model file: " + path);
    }

    // 第四步：从映射的起始 28 字节读配置；跳过头部后才是 FP32 权重。
    // 构造中一旦校验失败，C++ 不会调用本对象析构函数，因此要在此
    // 手动解除已创建的映射，避免异常路径泄漏地址空间。
    try {
        std::memcpy(&config_, mapping_, sizeof(config_));
        if (config_.dim <= 0 || config_.hidden_dim <= 0) {
            throw std::runtime_error("model dimensions must be positive");
        }

        const auto dim = static_cast<std::size_t>(config_.dim);
        const auto hidden_dim = static_cast<std::size_t>(config_.hidden_dim);
        if (dim > std::numeric_limits<std::size_t>::max() / hidden_dim) {
            throw std::overflow_error("model weight count overflow");
        }
        weight_count_ = dim * hidden_dim;
        if (weight_count_ >
            (size_bytes_ - sizeof(ModelConfig)) / sizeof(float)) {
            throw std::runtime_error(
                "model file ends inside the first FP32 weight tensor");
        }
    } catch (...) {
        ::munmap(mapping_, size_bytes_);
        mapping_ = nullptr;
        throw;
    }
}

MappedModelFile::~MappedModelFile() {
    if (mapping_ != nullptr) {
        ::munmap(mapping_, size_bytes_);
    }
}

float MappedModelFile::weight_at(std::size_t index) const {
    if (index >= weight_count_) {
        throw std::out_of_range("model weight index out of range");
    }

    float value;
    const std::byte* weight_bytes = data() + sizeof(ModelConfig);
    std::memcpy(&value, weight_bytes + index * sizeof(float), sizeof(value));
    return value;
}

} // namespace mini_infer
