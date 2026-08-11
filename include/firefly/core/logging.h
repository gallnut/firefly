#pragma once

#include <cstddef>
#include <format>
#include <optional>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>

namespace firefly::log
{

enum class Level : unsigned char
{
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Critical
};

enum class ColorMode : unsigned char
{
    Auto,
    Always,
    Never
};

struct Options
{
    Level     level = Level::Info;
    ColorMode color = ColorMode::Auto;
    bool      detailed = false;
    size_t    queue_capacity = 8192;
};

[[nodiscard]] std::optional<Level>     parse_level(std::string_view value) noexcept;
[[nodiscard]] std::optional<ColorMode> parse_color_mode(std::string_view value) noexcept;
[[nodiscard]] Options                  options_from_environment();

void configure(const Options& options);
[[nodiscard]] bool enabled(Level level) noexcept;
void submit(Level level, std::string_view component, std::string message, std::source_location location);
void submit_sync(Level level, std::string_view component, std::string message, std::source_location location);
void flush();
void shutdown();

template <typename... Args>
void write(Level level, std::string_view component, std::source_location location,
           std::format_string<Args...> format, Args&&... args)
{
    if (!enabled(level)) return;
    submit(level, component, std::format(format, std::forward<Args>(args)...), location);
}

template <typename... Args>
void write_sync(Level level, std::string_view component, std::source_location location,
                std::format_string<Args...> format, Args&&... args)
{
    if (!enabled(level)) return;
    submit_sync(level, component, std::format(format, std::forward<Args>(args)...), location);
}

}  // namespace firefly::log

#define FIREFLY_LOG_TRACE(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Trace, component, std::source_location::current(), __VA_ARGS__)
#define FIREFLY_LOG_DEBUG(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Debug, component, std::source_location::current(), __VA_ARGS__)
#define FIREFLY_LOG_INFO(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Info, component, std::source_location::current(), __VA_ARGS__)
#define FIREFLY_LOG_WARN(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Warn, component, std::source_location::current(), __VA_ARGS__)
#define FIREFLY_LOG_ERROR(component, ...)                                                                    \
    ::firefly::log::write(::firefly::log::Level::Error, component, std::source_location::current(), __VA_ARGS__)
#define FIREFLY_LOG_CRITICAL(component, ...)                                                                    \
    ::firefly::log::write_sync(::firefly::log::Level::Critical, component, std::source_location::current(),     \
                               __VA_ARGS__)
