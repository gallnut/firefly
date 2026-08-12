#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <string>

#include "firefly/device/error.h"

namespace firefly::device
{

/**
 * @brief RAII wrapper for CUDA Graph and Graph Execution.
 * Encapsulates graph capture, instantiation, and launching.
 */
class Graph
{
public:
    /** @brief Constructs an empty graph wrapper. */
    Graph() = default;

    /** @brief CUDA graph ownership cannot be copied. */
    Graph(const Graph&) = delete;
    /** @brief CUDA graph ownership cannot be copy-assigned. */
    Graph& operator=(const Graph&) = delete;

    /** @brief Transfers graph and executable ownership from another wrapper. */
    Graph(Graph&& other) noexcept : graph_(other.graph_), exec_(other.exec_)
    {
        other.graph_ = nullptr;
        other.exec_ = nullptr;
    }

    /** @brief Releases current resources and transfers ownership from another wrapper. */
    Graph& operator=(Graph&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            graph_ = other.graph_;
            exec_ = other.exec_;
            other.graph_ = nullptr;
            other.exec_ = nullptr;
        }
        return *this;
    }

    /** @brief Destroys both the executable graph and captured graph definition. */
    ~Graph() { destroy(); }

    /**
     * @brief Capture a sequence of CUDA operations into the graph.
     *
     * @param stream The stream to record operations on.
     * @param func A lambda or function executing the CUDA kernels/memcpys.
     * @return Result<void> Success or failure.
     */
    Result<void> capture(cudaStream_t stream, std::function<Status()> func)
    {
        // Destroy previous graph if any
        destroy();

        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess)
        {
            return unexpected(cuda_error(err, "begin CUDA stream capture"));
        }

        auto callback_status = func();
        if (!callback_status)
        {
            cudaGraph_t abandoned_graph = nullptr;
            cudaStreamEndCapture(stream, &abandoned_graph);
            if (abandoned_graph != nullptr) cudaGraphDestroy(abandoned_graph);
            return unexpected(std::move(callback_status.error()).with_context("execute CUDA graph capture callback"));
        }

        err = cudaStreamEndCapture(stream, &graph_);
        if (err != cudaSuccess)
        {
            return unexpected(cuda_error(err, "end CUDA stream capture"));
        }

        return instantiate();
    }

    /**
     * @brief Launches the instantiated graph.
     *
     * @param stream The stream to launch the graph in.
     * @return Result<void> Success or failure.
     */
    Result<void> launch(cudaStream_t stream) const
    {
        if (!exec_)
        {
            return unexpected(Error{ErrorCode::InvalidState, "CUDA graph executable has not been instantiated"});
        }

        cudaError_t err = cudaGraphLaunch(exec_, stream);
        if (err != cudaSuccess)
        {
            return unexpected(cuda_error(err, "launch CUDA graph"));
        }
        return {};
    }

    /** @brief Returns true when no graph definition has been captured. */
    [[nodiscard]] bool empty() const { return graph_ == nullptr; }

private:
    cudaGraph_t     graph_{nullptr}; ///< Owned captured graph definition.
    cudaGraphExec_t exec_{nullptr}; ///< Owned executable instantiated from `graph_`.

    /**
     * @brief Rebuilds the executable CUDA graph from the current graph definition.
     * @return Success, or a CUDA error when graph instantiation fails.
     */
    Result<void> instantiate()
    {
        if (exec_)
        {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }

        if (graph_)
        {
            // cudaGraphInstantiate creates an executable graph from a graph structure.
            cudaError_t err;
#if CUDART_VERSION >= 12000
            err = cudaGraphInstantiate(&exec_, graph_, 0);
#else
            err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
#endif

            if (err != cudaSuccess)
            {
                return unexpected(cuda_error(err, "instantiate CUDA graph"));
            }
        }
        return {};
    }

    /** @brief Releases owned executable and graph handles and restores the empty state. */
    void destroy()
    {
        if (exec_)
        {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_)
        {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }
    }

    // Helper 'check' removed
};

}  // namespace firefly::device
