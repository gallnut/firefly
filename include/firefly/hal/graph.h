#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <string>

#include "firefly/hal/error.h"

namespace firefly::hal
{

/**
 * @brief RAII wrapper for CUDA Graph and Graph Execution.
 * Encapsulates graph capture, instantiation, and launching.
 */
class Graph
{
public:
    Graph() = default;

    // Move-only
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;

    Graph(Graph&& other) noexcept : graph_(other.graph_), exec_(other.exec_)
    {
        other.graph_ = nullptr;
        other.exec_ = nullptr;
    }

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

    ~Graph() { destroy(); }

    /**
     * @brief Capture a sequence of CUDA operations into the graph.
     *
     * @param stream The stream to record operations on.
     * @param func A lambda or function executing the CUDA kernels/memcpys.
     * @return Result<void> Success or failure.
     */
    Result<void> capture(cudaStream_t stream, std::function<void()> func)
    {
        // Destroy previous graph if any
        destroy();

        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess)
        {
            return unexpected(Error{static_cast<int>(err), ErrorCategory::CUDA, "Failed to begin stream capture"});
        }

        try
        {
            func();
        }
        catch (const std::exception& e)
        {
            cudaStreamEndCapture(stream, &graph_);
            return unexpected(
                Error{-1, ErrorCategory::Runtime, "Exception during graph capture: " + std::string(e.what())});
        }
        catch (...)
        {
            cudaStreamEndCapture(stream, &graph_);
            return unexpected(Error{-1, ErrorCategory::Runtime, "Unknown exception during graph capture"});
        }

        err = cudaStreamEndCapture(stream, &graph_);
        if (err != cudaSuccess)
        {
            return unexpected(Error{static_cast<int>(err), ErrorCategory::CUDA, "Failed to end stream capture"});
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
            return unexpected(Error{-1, ErrorCategory::Runtime, "Graph: Execution not instantiated. Did you capture?"});
        }

        cudaError_t err = cudaGraphLaunch(exec_, stream);
        if (err != cudaSuccess)
        {
            return unexpected(Error{static_cast<int>(err), ErrorCategory::CUDA, "Failed to launch graph"});
        }
        return {};
    }

    [[nodiscard]] bool empty() const { return graph_ == nullptr; }

private:
    cudaGraph_t     graph_{nullptr};
    cudaGraphExec_t exec_{nullptr};

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
                return unexpected(Error{static_cast<int>(err), ErrorCategory::CUDA, "Failed to instantiate graph"});
            }
        }
        return {};
    }

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

}  // namespace firefly::hal
