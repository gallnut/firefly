#pragma once

#include <cstddef>
#include <format>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

#include "firefly/core/error.h"

namespace firefly::log
{

/** @brief Ordered severity levels accepted by the asynchronous logger. */
enum class Level : unsigned char
{
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical
};

/** @brief Controls whether terminal log output contains ANSI color sequences. */
enum class ColorMode : unsigned char
{
    Auto,
    Always,
    Never
};

/** @brief Runtime configuration for Firefly's process-wide logging backend. */
struct Options
{
    Level     level = Level::Info;       ///< Minimum severity submitted to the backend.
    ColorMode color = ColorMode::Auto;   ///< Terminal color policy.
    bool      detailed = false;          ///< Whether to include source file and line metadata.
    size_t    queue_capacity = 8192;     ///< Maximum number of pending asynchronous records.
};

/** @brief Parses a case-insensitive textual severity name. */
[[nodiscard]] std::optional<Level>     parse_level(std::string_view value) noexcept;
/** @brief Parses a textual color mode (`auto`, `always`, or `never`). */
[[nodiscard]] std::optional<ColorMode> parse_color_mode(std::string_view value) noexcept;
/** @brief Builds logger options from the supported `FIREFLY_LOG_*` environment variables. */
[[nodiscard]] Result<Options>          options_from_environment();

/** @brief Reconfigures the process-wide logger and its asynchronous queue. */
Status configure(const Options& options);
/** @brief Returns whether a record at `level` would currently be emitted. */
[[nodiscard]] bool enabled(Level level) noexcept;
/** @brief Enqueues a formatted record for asynchronous delivery. */
void submit(Level level, std::string_view component, std::string message, std::source_location location);
/** @brief Emits a record synchronously, bypassing the asynchronous queue. */
void submit_sync(Level level, std::string_view component, std::string message, std::source_location location);
/** @brief Blocks until all records submitted before the call have been written. */
void flush();
/** @brief Flushes records and stops the process-wide logging worker. */
void shutdown();

/**
 * @brief Formats and asynchronously submits a log record when its level is enabled.
 * @tparam Args Types consumed by the checked format string.
 * @param level Record severity.
 * @param component Short subsystem label.
 * @param location Call-site source location.
 * @param format Compile-time checked format string.
 * @param args Values interpolated into `format`.
 */
template <typename... Args>
void write(Level level, std::string_view component, std::source_location location,
           std::format_string<Args...> format, Args&&... args)
{
    if (!enabled(level)) return;
    submit(level, component, std::format(format, std::forward<Args>(args)...), location);
}

/**
 * @brief Formats and synchronously submits a log record when its level is enabled.
 * @copydetails write
 */
template <typename... Args>
void write_sync(Level level, std::string_view component, std::source_location location,
                std::format_string<Args...> format, Args&&... args)
{
    if (!enabled(level)) return;
    submit_sync(level, component, std::format(format, std::forward<Args>(args)...), location);
}

}  // namespace firefly::log

/**
 * @brief Submits a trace-level record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_TRACE(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Trace, component, std::source_location::current(), __VA_ARGS__)
/**
 * @brief Submits a debug-level record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_DEBUG(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Debug, component, std::source_location::current(), __VA_ARGS__)
/**
 * @brief Submits an informational record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_INFO(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Info, component, std::source_location::current(), __VA_ARGS__)
/**
 * @brief Submits a warning record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_WARN(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Warn, component, std::source_location::current(), __VA_ARGS__)
/**
 * @brief Submits an error record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_ERROR(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Error, component, std::source_location::current(), __VA_ARGS__)
/**
 * @brief Synchronously emits a critical record with the current source location.
 * @param component Short subsystem label.
 * @param ... Checked format string followed by its interpolation arguments.
 */
#define FIREFLY_LOG_CRITICAL(component, ...)                                                                    \
    ::firefly::log::write_sync(::firefly::log::Level::Critical, component, std::source_location::current(),     \
                               __VA_ARGS__)
