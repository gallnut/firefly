#include "firefly/service/server.h"

#include <atomic>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "firefly/execution/engine.h"
#include "firefly/execution/result_queue.h"
#include "firefly/service/session.h"
#include "firefly/model/tokenizer.h"
#include "firefly/service/grpc_adapter.h"

namespace firefly::service
{
namespace
{
std::vector<int> parse_prompt(const std::string& body, const model::Tokenizer& tokenizer, int& max_tokens)
{
    using json = nlohmann::json;
    std::vector<int> input_ids;
    int im_start_id = tokenizer.token_id("<|im_start|>");
    int im_end_id = tokenizer.token_id("<|im_end|>");
    max_tokens = 512;

    if (!body.empty())
    {
        auto request = json::parse(body);
        if (request.contains("max_tokens")) max_tokens = request["max_tokens"].get<int>();

        auto append_message = [&](const std::string& role, const std::string& content)
        {
            input_ids.push_back(im_start_id);
            auto role_ids = tokenizer.encode(role + "\n");
            input_ids.insert(input_ids.end(), role_ids.begin(), role_ids.end());
            auto content_ids = tokenizer.encode(content);
            input_ids.insert(input_ids.end(), content_ids.begin(), content_ids.end());
            input_ids.push_back(im_end_id);
            auto newline_ids = tokenizer.encode("\n");
            input_ids.insert(input_ids.end(), newline_ids.begin(), newline_ids.end());
        };

        if (request.contains("messages") && request["messages"].is_array())
        {
            for (const auto& message : request["messages"])
                append_message(message.value("role", "user"), message.value("content", ""));
        }
        else if (request.contains("prompt"))
        {
            append_message("user", request["prompt"].get<std::string>());
        }
    }

    input_ids.push_back(im_start_id);
    auto assistant_ids = tokenizer.encode("assistant\n");
    input_ids.insert(input_ids.end(), assistant_ids.begin(), assistant_ids.end());
    return input_ids;
}
}  // namespace

class Server::Impl
{
public:
    Impl(execution::Engine& engine, const model::Tokenizer& tokenizer, execution::ResultQueue& results)
        : engine_(engine), tokenizer_(tokenizer), results_(results)
    {
    }

    ~Impl() { stop_dispatcher(); }

    void run(int port)
    {
        start_dispatcher();
        GrpcAdapter adapter;
        adapter.start_server(port,
                             [this](const std::string& body, bool, std::shared_ptr<Session> session)
                             { submit(body, std::move(session)); });
        stop_dispatcher();
    }

private:
    void start_dispatcher()
    {
        dispatcher_running_ = true;
        dispatcher_thread_ = std::thread(
            [this]()
            {
                while (dispatcher_running_)
                {
                    auto item = results_.pop();
                    if (item.req_id.empty() && item.is_finished) break;

                    std::shared_ptr<Session> session;
                    {
                        std::lock_guard<std::mutex> lock(sessions_mutex_);
                        auto iterator = sessions_.find(item.req_id);
                        if (iterator != sessions_.end()) session = iterator->second;
                    }
                    if (!session) continue;

                    session->push(item.text, item.is_finished, item.has_usage ? item.prompt_tokens : -1,
                                  item.has_usage ? item.completion_tokens : -1);
                    if (item.is_finished)
                    {
                        std::lock_guard<std::mutex> lock(sessions_mutex_);
                        sessions_.erase(item.req_id);
                    }
                }
            });
    }

    void stop_dispatcher()
    {
        if (!dispatcher_thread_.joinable()) return;
        dispatcher_running_ = false;
        results_.push("", "", true);
        dispatcher_thread_.join();
    }

    void submit(const std::string& body, std::shared_ptr<Session> session)
    {
        try
        {
            int max_tokens = 0;
            auto input_ids = parse_prompt(body, tokenizer_, max_tokens);
            auto request_id = "req-" + std::to_string(next_request_id_.fetch_add(1));
            {
                std::lock_guard<std::mutex> lock(sessions_mutex_);
                sessions_[request_id] = session;
            }
            std::cout << "[Server] Request chat completions limit: " << max_tokens << std::endl;
            engine_.async_generate(request_id, input_ids, max_tokens, session->cancel_flag);
        }
        catch (const std::exception& error)
        {
            nlohmann::json response;
            response["error"] = {{"message", error.what()}, {"type", "server_error"}};
            session->push(response.dump(), true);
        }
    }

    execution::Engine& engine_;
    const model::Tokenizer& tokenizer_;
    execution::ResultQueue& results_;
    std::mutex sessions_mutex_;
    std::unordered_map<std::string, std::shared_ptr<Session>> sessions_;
    std::atomic<uint64_t> next_request_id_{0};
    std::atomic<bool> dispatcher_running_{false};
    std::thread dispatcher_thread_;
};

Server::Server(execution::Engine& engine, const model::Tokenizer& tokenizer, execution::ResultQueue& results)
    : impl_(std::make_unique<Impl>(engine, tokenizer, results))
{
}

Server::~Server() = default;

void Server::run(int port) { impl_->run(port); }

}  // namespace firefly::service
