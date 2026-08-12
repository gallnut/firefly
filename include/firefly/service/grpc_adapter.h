#pragma once

#include <functional>
#include <memory>
#include <string>

#include "firefly/core/error.h"
#include "firefly/service/session.h"

namespace firefly
{
namespace service
{

/** @brief Thin adapter that exposes JSON generation requests through the generated gRPC service. */
class GrpcAdapter
{
public:
    /** @brief Constructs an idle adapter. */
    GrpcAdapter() = default;
    /** @brief Releases server-side resources after `start_server` exits. */
    ~GrpcAdapter() = default;

    /**
     * @brief Starts a blocking gRPC server bound to all interfaces on `port`.
     * @param port TCP port to bind.
     * @param handler Callback receiving request JSON, streaming mode, and a shared response session.
     * @return Success when the server starts and exits normally, or a structured bind/setup error.
     * @note The call blocks until the gRPC server shuts down.
     */
    Status start_server(int port, std::function<void(const std::string&, bool, std::shared_ptr<Session>)> handler);
};

}  // namespace service
}  // namespace firefly
