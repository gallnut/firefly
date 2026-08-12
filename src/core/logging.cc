#include "firefly/core/logging.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <deque>
#include <functional>
#include <mutex>
#include <print>
#include <stdexcept>
#include <thread>
#include <unistd.h>

namespace firefly::log
{
namespace
{
using Clock = std::chrono::system_clock;

struct Record
{
    Clock::time_point     timestamp;
    Level                 level;
    std::string           component;
    std::string           message;
    std::source_location  location;
    size_t                 thread_id;
};

constexpr std::string_view reset = "\033[0m";
constexpr std::string_view dim = "\033[2m";
constexpr std::string_view component_color = "\033[36m";

std::string_view level_name(Level level)
{
    switch (level)
    {
        case Level::Trace: return "TRACE";
        case Level::Debug: return "DEBUG";
        case Level::Info: return "INFO";
        case Level::Warn: return "WARN";
        case Level::Error: return "ERROR";
        case Level::Critical: return "FATAL";
    }
    return "UNKNOWN";
}

std::string_view level_color(Level level)
{
    switch (level)
    {
        case Level::Trace: return "\033[90m";
        case Level::Debug: return "\033[34m";
        case Level::Info: return "\033[32m";
        case Level::Warn: return "\033[33m";
        case Level::Error: return "\033[31m";
        case Level::Critical: return "\033[1;97;41m";
    }
    return {};
}

std::string_view relative_source_path(std::string_view path)
{
    constexpr std::string_view roots[] = {"/src/", "/include/", "/tests/"};
    for (std::string_view root : roots)
    {
        if (size_t position = path.rfind(root); position != std::string_view::npos) return path.substr(position + 1);
    }
    return path;
}

bool environment_flag(std::string_view name, bool default_value)
{
    const char* value = std::getenv(std::string(name).c_str());
    if (value == nullptr || *value == '\0') return default_value;
    std::string_view text(value);
    return text != "0" && text != "false" && text != "off" && text != "no";
}

class Logger
{
public:
    Logger() : worker_([this] { run(); }) {}
    ~Logger() { shutdown(); }

    bool configure(const Options& options)
    {
        if (options.queue_capacity == 0) return false;
        std::lock_guard lock(queue_mutex_);
        options_ = options;
        minimum_level_.store(options.level, std::memory_order_relaxed);
        detailed_.store(options.detailed, std::memory_order_relaxed);
        color_.store(options.color, std::memory_order_relaxed);
        return true;
    }

    bool enabled(Level level) const noexcept
    {
        return level >= minimum_level_.load(std::memory_order_relaxed);
    }

    void submit(Record record)
    {
        std::unique_lock lock(queue_mutex_);
        if (stopping_)
        {
            lock.unlock();
            emit(record);
            return;
        }

        if (queue_.size() >= options_.queue_capacity)
        {
            if (record.level < Level::Warn)
            {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return;
            }
            space_available_.wait(lock, [this] { return stopping_ || queue_.size() < options_.queue_capacity; });
            if (stopping_)
            {
                lock.unlock();
                emit(record);
                return;
            }
        }

        queue_.push_back(std::move(record));
        record_available_.notify_one();
    }

    void submit_sync(const Record& record)
    {
        flush();
        emit(record);
        std::fflush(stderr);
    }

    void flush()
    {
        std::unique_lock lock(queue_mutex_);
        drained_.wait(lock, [this] { return queue_.empty() && !writing_; });
        lock.unlock();
        std::lock_guard output_lock(output_mutex_);
        std::fflush(stdout);
        std::fflush(stderr);
    }

    void shutdown()
    {
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_) return;
            stopping_ = true;
        }
        record_available_.notify_all();
        space_available_.notify_all();
        if (worker_.joinable()) worker_.join();
        std::lock_guard output_lock(output_mutex_);
        std::fflush(stdout);
        std::fflush(stderr);
    }

private:
    std::string format_timestamp(Clock::time_point timestamp) const
    {
        const auto seconds = std::chrono::time_point_cast<std::chrono::seconds>(timestamp);
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(timestamp - seconds).count();
        const std::time_t value = Clock::to_time_t(timestamp);
        std::tm           local_time{};
        localtime_r(&value, &local_time);
        return std::format("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}", local_time.tm_year + 1900,
                           local_time.tm_mon + 1, local_time.tm_mday, local_time.tm_hour, local_time.tm_min,
                           local_time.tm_sec, milliseconds);
    }

    bool use_color(FILE* stream) const
    {
        switch (color_.load(std::memory_order_relaxed))
        {
            case ColorMode::Always: return true;
            case ColorMode::Never: return false;
            case ColorMode::Auto: return ::isatty(::fileno(stream)) != 0 && std::getenv("NO_COLOR") == nullptr;
        }
        return false;
    }

    std::string format_record(const Record& record, bool colored) const
    {
        const std::string timestamp = format_timestamp(record.timestamp);
        const std::string_view level = level_name(record.level);
        const std::string_view source = relative_source_path(record.location.file_name());
        std::string detail;
        if (detailed_.load(std::memory_order_relaxed))
        {
            detail = std::format(" [tid={:x} {}:{} {}]", record.thread_id, source, record.location.line(),
                                 record.location.function_name());
        }

        if (!colored)
        {
            return std::format("{} {:<5} {:<10}{} {}", timestamp, level, record.component, detail, record.message);
        }

        return std::format("{}{}{} {}{:<5}{} {}{:<10}{}{}{}{} {}", dim, timestamp, reset,
                           level_color(record.level), level, reset, component_color, record.component, reset, dim,
                           detail, reset, record.message);
    }

    void emit(const Record& record) const
    {
        FILE* stream = record.level >= Level::Error ? stderr : stdout;
        const std::string line = format_record(record, use_color(stream));
        std::lock_guard output_lock(output_mutex_);
        std::println(stream, "{}", line);
        std::fflush(stream);
    }

    void emit_dropped(size_t count)
    {
        if (count == 0) return;
        Record record{Clock::now(), Level::Warn, "logging", std::format("asynchronous queue overflow dropped={}", count),
                      std::source_location::current(), std::hash<std::thread::id>{}(std::this_thread::get_id())};
        emit(record);
    }

    void run()
    {
        while (true)
        {
            Record record;
            {
                std::unique_lock lock(queue_mutex_);
                record_available_.wait(lock, [this] { return stopping_ || !queue_.empty(); });
                if (queue_.empty() && stopping_) break;
                record = std::move(queue_.front());
                queue_.pop_front();
                writing_ = true;
                space_available_.notify_one();
            }

            emit_dropped(dropped_.exchange(0, std::memory_order_relaxed));
            emit(record);

            {
                std::lock_guard lock(queue_mutex_);
                writing_ = false;
                if (queue_.empty()) drained_.notify_all();
            }
        }
        emit_dropped(dropped_.exchange(0, std::memory_order_relaxed));
        std::lock_guard lock(queue_mutex_);
        writing_ = false;
        drained_.notify_all();
    }

    Options              options_;
    std::atomic<Level>    minimum_level_{Level::Info};
    std::atomic<bool>     detailed_{false};
    std::atomic<ColorMode> color_{ColorMode::Auto};
    mutable std::mutex    output_mutex_;
    std::mutex            queue_mutex_;
    std::condition_variable record_available_;
    std::condition_variable space_available_;
    std::condition_variable drained_;
    std::deque<Record>    queue_;
    std::atomic<size_t>    dropped_{0};
    bool                   writing_ = false;
    bool                   stopping_ = false;
    std::thread            worker_;
};

Logger& logger()
{
    static Logger instance;
    return instance;
}

Record make_record(Level level, std::string_view component, std::string message, std::source_location location)
{
    return {Clock::now(), level, std::string(component), std::move(message), location,
            std::hash<std::thread::id>{}(std::this_thread::get_id())};
}
}  // namespace

std::optional<Level> parse_level(std::string_view value) noexcept
{
    if (value == "trace") return Level::Trace;
    if (value == "debug") return Level::Debug;
    if (value == "info") return Level::Info;
    if (value == "warn" || value == "warning") return Level::Warn;
    if (value == "error") return Level::Error;
    if (value == "critical" || value == "fatal") return Level::Critical;
    return std::nullopt;
}

std::optional<ColorMode> parse_color_mode(std::string_view value) noexcept
{
    if (value == "auto") return ColorMode::Auto;
    if (value == "always" || value == "on") return ColorMode::Always;
    if (value == "never" || value == "off") return ColorMode::Never;
    return std::nullopt;
}

Result<Options> options_from_environment()
{
    Options options;
    if (const char* value = std::getenv("FIREFLY_LOG_LEVEL"))
    {
        auto level = parse_level(value);
        if (!level) return unexpected(Error{ErrorCode::InvalidArgument, "invalid FIREFLY_LOG_LEVEL"});
        options.level = *level;
    }
    if (const char* value = std::getenv("FIREFLY_LOG_COLOR"))
    {
        auto color = parse_color_mode(value);
        if (!color) return unexpected(Error{ErrorCode::InvalidArgument, "invalid FIREFLY_LOG_COLOR"});
        options.color = *color;
    }
    options.detailed = environment_flag("FIREFLY_LOG_DETAIL", false);
    return options;
}

Status configure(const Options& options)
{
    if (!logger().configure(options))
        return unexpected(Error{ErrorCode::InvalidArgument, "log queue capacity must be positive"});
    return {};
}

bool enabled(Level level) noexcept { return logger().enabled(level); }

void submit(Level level, std::string_view component, std::string message, std::source_location location)
{
    logger().submit(make_record(level, component, std::move(message), location));
}

void submit_sync(Level level, std::string_view component, std::string message, std::source_location location)
{
    logger().submit_sync(make_record(level, component, std::move(message), location));
}

void flush() { logger().flush(); }

void shutdown() { logger().shutdown(); }

}  // namespace firefly::log
