#pragma once

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>

#if defined(__linux__)
    #include <sys/ioctl.h>
    #include <sys/syscall.h>
    #include <unistd.h>
    #include <linux/perf_event.h>
#endif

namespace atlas::hpc {

struct HardwareMetrics {
    double wall_seconds{0.0};
    std::optional<double> energy_joules;
    std::uint64_t cpu_cycles{0};
    std::uint64_t instructions{0};
    std::uint64_t llc_misses{0};

    [[nodiscard]] double ipc() const noexcept {
        return cpu_cycles == 0 ? 0.0 : static_cast<double>(instructions) / static_cast<double>(cpu_cycles);
    }
};

namespace detail {

using Clock = std::chrono::steady_clock;

inline std::optional<std::uint64_t> read_rapl_energy_uj() noexcept {
#if defined(__linux__)
    const char* paths[] = {
        "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
        "/sys/devices/virtual/powercap/intel-rapl/intel-rapl:0/energy_uj"
    };
    for (const char* path : paths) {
        std::ifstream file(path);
        if (!file.is_open()) continue;
        std::uint64_t value = 0;
        file >> value;
        if (file) return value;
    }
#endif
    return std::nullopt;
}

#if defined(__linux__)
inline long perf_event_open(perf_event_attr* attr, pid_t pid, int cpu, int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

inline bool read_counter(int fd, std::uint64_t& value) noexcept {
    if (fd < 0) return false;
    const auto bytes = ::read(fd, &value, sizeof(value));
    return bytes == static_cast<ssize_t>(sizeof(value));
}
#endif

} // namespace detail

class ScopedHardwareProfiler {
public:
    ScopedHardwareProfiler() = default;

    void start() noexcept {
        start_time_ = detail::Clock::now();
        energy_start_ = detail::read_rapl_energy_uj();

#if defined(__linux__)
        open_counters();
        if (cycles_fd_ >= 0) {
            const std::uint64_t zero = 0;
            (void)::ioctl(cycles_fd_, PERF_EVENT_IOC_RESET, 0);
            (void)::ioctl(cycles_fd_, PERF_EVENT_IOC_ENABLE, 0);
            (void)::write(cycles_fd_, &zero, 0);
        }
        if (instructions_fd_ >= 0) {
            (void)::ioctl(instructions_fd_, PERF_EVENT_IOC_RESET, 0);
            (void)::ioctl(instructions_fd_, PERF_EVENT_IOC_ENABLE, 0);
        }
        if (llc_misses_fd_ >= 0) {
            (void)::ioctl(llc_misses_fd_, PERF_EVENT_IOC_RESET, 0);
            (void)::ioctl(llc_misses_fd_, PERF_EVENT_IOC_ENABLE, 0);
        }
#endif
    }

    [[nodiscard]] HardwareMetrics stop() noexcept {
        HardwareMetrics metrics;
        const auto stop_time = detail::Clock::now();
        metrics.wall_seconds = std::chrono::duration<double>(stop_time - start_time_).count();

        const auto end_energy = detail::read_rapl_energy_uj();
        if (energy_start_ && end_energy && *end_energy >= *energy_start_) {
            metrics.energy_joules = static_cast<double>(*end_energy - *energy_start_) * 1.0e-6;
        }

#if defined(__linux__)
        if (cycles_fd_ >= 0) {
            (void)::ioctl(cycles_fd_, PERF_EVENT_IOC_DISABLE, 0);
            (void)detail::read_counter(cycles_fd_, metrics.cpu_cycles);
            ::close(cycles_fd_);
            cycles_fd_ = -1;
        }
        if (instructions_fd_ >= 0) {
            (void)::ioctl(instructions_fd_, PERF_EVENT_IOC_DISABLE, 0);
            (void)detail::read_counter(instructions_fd_, metrics.instructions);
            ::close(instructions_fd_);
            instructions_fd_ = -1;
        }
        if (llc_misses_fd_ >= 0) {
            (void)::ioctl(llc_misses_fd_, PERF_EVENT_IOC_DISABLE, 0);
            (void)detail::read_counter(llc_misses_fd_, metrics.llc_misses);
            ::close(llc_misses_fd_);
            llc_misses_fd_ = -1;
        }
#endif

        return metrics;
    }

private:
#if defined(__linux__)
    void open_counters() noexcept {
        if (counters_opened_) return;
        counters_opened_ = true;

        perf_event_attr attr{};
        attr.size = sizeof(attr);
        attr.disabled = 1;
        attr.exclude_kernel = 0;
        attr.exclude_hv = 0;
        attr.inherit = 1;
        attr.read_format = 0;

        attr.type = PERF_TYPE_HARDWARE;
        attr.config = PERF_COUNT_HW_CPU_CYCLES;
        cycles_fd_ = static_cast<int>(detail::perf_event_open(&attr, 0, -1, -1, 0));

        attr.config = PERF_COUNT_HW_INSTRUCTIONS;
        instructions_fd_ = static_cast<int>(detail::perf_event_open(&attr, 0, -1, -1, 0));

        attr.type = PERF_TYPE_HW_CACHE;
        attr.config = (PERF_COUNT_HW_CACHE_LL)
                    | (PERF_COUNT_HW_CACHE_OP_READ << 8)
                    | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
        llc_misses_fd_ = static_cast<int>(detail::perf_event_open(&attr, 0, -1, -1, 0));
    }

    bool counters_opened_{false};
    int cycles_fd_{-1};
    int instructions_fd_{-1};
    int llc_misses_fd_{-1};
#endif

    detail::Clock::time_point start_time_{};
    std::optional<std::uint64_t> energy_start_;
};

} // namespace atlas::hpc
