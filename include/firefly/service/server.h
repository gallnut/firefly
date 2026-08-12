#pragma once

#include <memory>

#include "firefly/core/error.h"

namespace firefly
{
namespace model
{
/** @brief Forward declaration of the tokenizer borrowed by the service facade. */
class Tokenizer;
}
namespace execution
{
/** @brief Forward declaration of the asynchronous inference engine. */
class Engine;
/** @brief Forward declaration of the engine-to-service result queue. */
class ResultQueue;
}
namespace service
{

/** @brief High-level JSON/gRPC serving facade around an engine, tokenizer, and result queue. */
class Server
{
public:
    /**
     * @brief Constructs a service borrowing runtime objects that must outlive it.
     * @param engine Running inference engine receiving parsed generation requests.
     * @param tokenizer Loaded tokenizer used for request encoding and stop-token handling.
     * @param results Queue from which streamed engine results are routed to sessions.
     */
    Server(execution::Engine& engine, const model::Tokenizer& tokenizer, execution::ResultQueue& results);
    /** @brief Stops internal response routing and releases the hidden implementation. */
    ~Server();

    /** @brief Server ownership cannot be copied. */
    Server(const Server&) = delete;
    /** @brief Server ownership cannot be copy-assigned. */
    Server& operator=(const Server&) = delete;

    /**
     * @brief Runs the blocking gRPC accept loop on the requested TCP port.
     * @param port Local TCP port bound by the service adapter.
     */
    Status run(int port);

private:
    /** @brief Opaque implementation isolating gRPC and JSON dependencies from public callers. */
    class Impl;
    std::unique_ptr<Impl> impl_; ///< Owned hidden transport and response-routing implementation.
};

}  // namespace service
}  // namespace firefly
