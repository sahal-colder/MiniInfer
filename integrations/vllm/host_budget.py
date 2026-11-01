"""Host KV budget, including both staging buffers and allocator rounding."""


def pool_allocation_bytes(size):
    """Existing cpu_detail::size_class rule (also used by the Pinned pool)."""
    if type(size) is not int or not 0 < size <= (1 << 64) - 64:
        raise ValueError("Invalid host allocation size")
    step = 64 if size <= 1024 else 1 << ((size - 1).bit_length() - 3)
    return (size + step - 1) // step * step


def pool_usable_budget(total):
    pool_allocation_bytes(total)  # Validate before shifting.
    if total <= 1024:
        return total // 64 * 64
    step = 1 << (total.bit_length() - 3)
    return total // step * step


def kv_slot_bytes(payload, compact=False):
    pool_allocation_bytes(payload)  # Validate, including overflow.
    if compact:
        return pool_allocation_bytes((payload + 4095) // 4096 * 4096)
    return max(4096, 1 << (payload - 1).bit_length())


def host_cache_budget(total, payload, buffer_slots, pooled, compact=False, device_reserved=0):
    if type(device_reserved) is not int or device_reserved < 0:
        raise ValueError("Invalid SSD device reservation")
    if any(type(n) is not int or n <= 0 for n in (total, payload, buffer_slots)):
        raise ValueError("Host budget, payload and buffer slots must be positive integers")
    if buffer_slots < 2 or buffer_slots % 2:
        raise ValueError("Staging requires an even number of slots >= 2")
    buffer = payload * buffer_slots
    if pooled:
        buffer = pool_allocation_bytes(buffer)
    else:
        # Torch's caching host allocator rounds to powers of two by default.
        # This remains a safe upper bound when caching/rounding is disabled.
        buffer = 1 << (buffer - 1).bit_length()
    remaining = total - 2 * buffer - device_reserved
    slot = kv_slot_bytes(payload, compact)
    if remaining < slot:
        raise ValueError("Total host budget cannot hold staging and one CPU slot")
    return remaining
