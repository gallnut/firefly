#pragma once

#include <functional>
#include <memory>
#include <string>

#include "firefly/session.h"

namespace firefly
{
namespace server
{

class GrpcAdapter
{
public:
    GrpcAdapter() = default;
    ~GrpcAdapter() = default;

    // Starts the gRPC server on `0.0.0.0:port`.
    // The handler takes a JSON string as input, a boolean indicating if it's a stream,
    // and a Session queue to append logic text into
    bool start_server(int port, std::function<void(const std::string&, bool, std::shared_ptr<Session>)> handler);
};

}  // namespace server
}  // namespace firefly
