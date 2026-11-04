#include "infer/io/safetensors.h"
#include "yyjson.h"
#include <algorithm>
#include <array>
#include <cerrno>
#include <fcntl.h>
#include <limits>
#include <set>
#include <stdexcept>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace infer::io {
DirectAlignment direct_alignment(int fd) {
#ifdef STATX_DIOALIGN
    struct statx result{};
    if (::statx(fd,"",AT_EMPTY_PATH,STATX_DIOALIGN,&result)==0 && (result.stx_mask & STATX_DIOALIGN)) {
        const auto bytes=std::max({page_bytes,std::size_t(result.stx_dio_mem_align),std::size_t(result.stx_dio_offset_align)});
        align_up(0,bytes);
        return {bytes,true};
    }
#endif
    // Older build headers/filesystems cannot report DIOALIGN. Requests still
    // use 4 KiB. The application never clears O_DIRECT; deployments with
    // unknown constraints must verify the filesystem's actual behavior.
    (void)fd;
    return {};
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    if (!alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("Alignment must be a power of two");
    if (value > SIZE_MAX - (alignment - 1)) throw std::overflow_error("Alignment overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}
AlignedRange aligned_range(std::uint64_t offset, std::size_t bytes, std::size_t alignment) {
    align_up(0, alignment);
    const auto prefix = std::size_t(offset % alignment);
    if (bytes > SIZE_MAX - prefix || offset > INT64_MAX || bytes > INT64_MAX - offset)
        throw std::overflow_error("File extent overflow");
    const auto rounded=align_up(prefix+bytes,alignment);
    if (rounded>std::uint64_t(INT64_MAX)-(offset-prefix)) throw std::overflow_error("Aligned file extent overflow");
    return {offset-prefix,rounded,prefix};
}
namespace {
void read_exact(int fd, void* destination, std::size_t bytes, std::uint64_t offset) {
    auto* data = static_cast<char*>(destination);
    while (bytes) {
        const auto n = pread(fd, data, bytes, offset);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) throw std::system_error(errno, std::generic_category(), "Safetensors header read");
        if (!n) throw std::runtime_error("Truncated Safetensors header");
        data += n; offset += n; bytes -= n;
    }
}
std::string_view view(yyjson_val* value) {
    if (!yyjson_is_str(value)) throw std::invalid_argument("Expected JSON string");
    return {yyjson_get_str(value), yyjson_get_len(value)};
}
std::size_t integer(yyjson_val* value) {
    if (!yyjson_is_uint(value) || yyjson_get_uint(value) > SIZE_MAX)
        throw std::invalid_argument("Expected unsigned integer");
    return yyjson_get_uint(value);
}
void unique_keys(yyjson_val* object) {
    if (!yyjson_is_obj(object)) throw std::invalid_argument("Expected JSON object");
    std::set<std::string_view> names;
    auto iter = yyjson_obj_iter_with(object);
    while (auto* key = yyjson_obj_iter_next(&iter))
        if (!names.insert(view(key)).second) throw std::invalid_argument("Duplicate JSON key");
}
std::size_t dtype_bits(std::string_view dtype) {
    if (dtype == "BOOL" || dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2") return 8;
    if (dtype == "F16" || dtype == "BF16" || dtype == "I16" || dtype == "U16") return 16;
    if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 32;
    if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 64;
    throw std::invalid_argument("Unsupported Safetensors dtype: " + std::string(dtype));
}
}
struct Safetensors::Impl {
    int fd = -1;
    std::uint64_t size = 0;
    std::vector<char> header;
    std::vector<TensorInfo> tensors;
    std::vector<std::pair<std::string_view,std::string_view>> metadata;
    ~Impl() { if (fd >= 0) close(fd); }
};
Safetensors::Safetensors(const std::string& path, std::size_t max_header_bytes)
    : impl_(std::make_unique<Impl>()) {
    impl_->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl_->fd < 0) throw std::system_error(errno, std::generic_category(), "Open Safetensors");
    struct stat st{};
    if (fstat(impl_->fd, &st)) throw std::system_error(errno, std::generic_category(), "Stat Safetensors");
    if (!S_ISREG(st.st_mode) || st.st_size < 8) throw std::invalid_argument("Invalid Safetensors file");
    impl_->size = st.st_size;
    std::array<unsigned char, 8> length{};
    read_exact(impl_->fd, length.data(), length.size(), 0);
    std::uint64_t n = 0;
    for (std::size_t i = 0; i < length.size(); ++i) n |= std::uint64_t(length[i]) << (8 * i);
    if (!n || n > max_header_bytes || n > impl_->size - 8 || n > SIZE_MAX - YYJSON_PADDING_SIZE)
        throw std::invalid_argument("Invalid Safetensors header length");
    impl_->header.resize(n + YYJSON_PADDING_SIZE);
    read_exact(impl_->fd, impl_->header.data(), n, 8);
    if (impl_->header[0] != '{') throw std::invalid_argument("Safetensors header must start with object");
    yyjson_read_err error{};
    std::unique_ptr<yyjson_doc, decltype(&yyjson_doc_free)> doc(
        yyjson_read_opts(impl_->header.data(), n, YYJSON_READ_INSITU, nullptr, &error), yyjson_doc_free);
    if (!doc) throw std::invalid_argument("Invalid Safetensors JSON: " + std::string(error.msg));
    auto* root = yyjson_doc_get_root(doc.get());
    unique_keys(root);
    auto iter = yyjson_obj_iter_with(root);
    while (auto* key = yyjson_obj_iter_next(&iter)) {
        auto* object = yyjson_obj_iter_get_val(key);
        unique_keys(object);
        if (view(key) == "__metadata__") {
            auto meta = yyjson_obj_iter_with(object);
            while (auto* k = yyjson_obj_iter_next(&meta)) impl_->metadata.emplace_back(view(k),view(yyjson_obj_iter_get_val(k)));
            continue;
        }
        if (yyjson_obj_size(object) != 3) throw std::invalid_argument("Unexpected tensor metadata fields");
        TensorInfo tensor;
        tensor.name = view(key);
        tensor.dtype = view(yyjson_obj_get(object, "dtype"));
        auto* shape = yyjson_obj_get(object, "shape");
        auto* offsets = yyjson_obj_get(object, "data_offsets");
        if (!yyjson_is_arr(shape) || !yyjson_is_arr(offsets) || yyjson_arr_size(offsets) != 2)
            throw std::invalid_argument("Invalid Safetensors shape/offsets");
        std::size_t index, count;
        yyjson_val* dimension;
        yyjson_arr_foreach(shape, index, count, dimension) tensor.shape.push_back(integer(dimension));
        std::size_t elements = 1;
        if (std::find(tensor.shape.begin(), tensor.shape.end(), 0) != tensor.shape.end()) elements = 0;
        if (elements) for (auto d : tensor.shape) {
            if (d > SIZE_MAX / elements) throw std::overflow_error("Tensor shape overflow");
            elements *= d;
        }
        const auto unit = dtype_bits(tensor.dtype) / 8;
        if (elements > SIZE_MAX / unit) throw std::overflow_error("Tensor bytes overflow");
        tensor.bytes = elements * unit;
        const auto begin = integer(yyjson_arr_get(offsets, 0)), end = integer(yyjson_arr_get(offsets, 1));
        if (end < begin || end - begin != tensor.bytes || end > impl_->size - 8 - n)
            throw std::invalid_argument("Safetensors tensor extent mismatch");
        tensor.file_offset = 8 + n + begin;
        impl_->tensors.push_back(std::move(tensor));
    }
    std::sort(impl_->tensors.begin(), impl_->tensors.end(), [](const auto& a, const auto& b) {
        return a.file_offset != b.file_offset ? a.file_offset < b.file_offset : a.bytes < b.bytes;
    });
    std::uint64_t end = 8 + n;
    for (const auto& tensor : impl_->tensors) {
        if (tensor.file_offset != end) throw std::invalid_argument("Safetensors overlap or unindexed hole");
        end += tensor.bytes;
    }
    if (end != impl_->size) throw std::invalid_argument("Safetensors has unindexed trailing bytes");
}
Safetensors::~Safetensors() = default;
int Safetensors::fd() const noexcept { return impl_->fd; }
std::uint64_t Safetensors::file_bytes() const noexcept { return impl_->size; }
const std::vector<TensorInfo>& Safetensors::tensors() const noexcept { return impl_->tensors; }
const TensorInfo& Safetensors::tensor(std::string_view name) const {
    for (const auto& tensor : impl_->tensors) if (tensor.name == name) return tensor;
    throw std::out_of_range("Unknown Safetensors tensor: " + std::string(name));
}
std::string_view Safetensors::metadata(std::string_view name) const {
    for (const auto& entry:impl_->metadata) if (entry.first==name) return entry.second;
    throw std::out_of_range("Missing Safetensors metadata: "+std::string(name));
}
} // namespace infer::io
