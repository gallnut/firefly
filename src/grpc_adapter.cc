#include "grpc_adapter.h"

#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>

#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>

#include "firefly.grpc.pb.h"
#include "firefly.pb.h"
#include "firefly/session.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerWriter;
using grpc::Status;
using json = nlohmann::json;

namespace firefly
{
namespace server
{

static std::string scrub_utf8(const std::string &str)
{
    std::string out;
    out.reserve(str.length());
    size_t i = 0;
    while (i < str.length())
    {
        unsigned char c = str[i];
        if (c < 0x80)
        {
            out += c;
            i++;
            continue;
        }

        size_t len = 0;
        if ((c & 0xE0) == 0xC0)
            len = 2;
        else if ((c & 0xF0) == 0xE0)
            len = 3;
        else if ((c & 0xF8) == 0xF0)
            len = 4;

        if (len == 0 || i + len > str.length())
        {
            i++;
            continue;
        }

        bool valid = true;
        for (size_t j = 1; j < len; ++j)
        {
            if ((str[i + j] & 0xC0) != 0x80)
            {
                valid = false;
                break;
            }
        }

        if (valid)
        {
            out.append(str, i, len);
            i += len;
        }
        else
        {
            i++;
        }
    }
    return out;
}

class InferenceServiceImpl final : public InferenceService::Service
{
public:
    InferenceServiceImpl(std::function<void(const std::string &, bool, std::shared_ptr<Session>)> handler)
        : handler_(std::move(handler))
    {
    }

    Status ChatCompletion(ServerContext *context, const ChatCompletionRequest *request,
                          ChatCompletionResponse *reply) override
    {
        json req_json;
        if (request->max_tokens() > 0) req_json["max_tokens"] = request->max_tokens();
        if (!request->model().empty()) req_json["model"] = request->model();

        if (request->messages_size() > 0)
        {
            json messages = json::array();
            for (int i = 0; i < request->messages_size(); ++i)
            {
                auto &msg = request->messages(i);
                messages.push_back({{"role", msg.role()}, {"content", msg.content()}});
            }
            req_json["messages"] = messages;
        }
        else if (!request->prompt().empty())
        {
            req_json["prompt"] = request->prompt();
        }

        std::string req_body = req_json.dump();

        auto session = std::make_shared<Session>();

        // We run the handler synchronously here (it returns instantly because it just maps req_id & engine.push)
        try
        {
            handler_(req_body, false, session);
        }
        catch (const std::exception &e)
        {
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }

        std::string full_response;

        // Block on the result queue waiting for the engine text buffer to construct the final response.
        std::unique_lock<std::mutex> lock(session->mtx);
        // We wait until is_finished, since this is a unary RPC, we just accumulate everything.
        session->cv.wait(lock, [&]() { return session->is_finished; });

        while (!session->output_queue.empty())
        {
            full_response += session->output_queue.front().first;
            session->output_queue.pop();
        }

        try
        {
            reply->set_id("chatcmpl-firefly");
            reply->set_object("chat.completion");
            reply->set_created(std::time(nullptr));
            reply->set_model(req_json.value("model", "qwen3"));

            // Parse Chain of Thought thinking blocks
            std::string content = full_response;
            std::string reasoning_content = "";

            std::string think_start = "<think>";
            std::string think_end = "</think>";

            size_t start_pos = content.find(think_start);
            if (start_pos != std::string::npos)
            {
                size_t end_pos = content.find(think_end, start_pos + think_start.length());
                if (end_pos != std::string::npos)
                {
                    reasoning_content =
                        content.substr(start_pos + think_start.length(), end_pos - (start_pos + think_start.length()));
                    content = content.substr(end_pos + think_end.length());
                }
                else
                {
                    reasoning_content = content.substr(start_pos + think_start.length());
                    content = "";
                }

                auto trim = [](std::string &s)
                {
                    s.erase(0, s.find_first_not_of(" \n\r\t"));
                    s.erase(s.find_last_not_of(" \n\r\t") + 1);
                };
                trim(reasoning_content);
                trim(content);
            }

            auto *choice = reply->add_choices();
            choice->set_index(0);
            choice->set_finish_reason("stop");

            auto *msg = choice->mutable_message();
            msg->set_role("assistant");
            msg->set_content(scrub_utf8(content));
            if (!reasoning_content.empty())
            {
                choice->set_reasoning_content(scrub_utf8(reasoning_content));
            }

            // Fake usage for now just as placeholder (since we didn't track prompt tokens here)
            auto *usage = reply->mutable_usage();
            usage->set_prompt_tokens(0);
            usage->set_completion_tokens(0);
            usage->set_total_tokens(0);
        }
        catch (const std::exception &e)
        {
            return Status(grpc::StatusCode::INTERNAL, e.what());
        }

        return Status::OK;
    }

    Status ChatCompletionStream(ServerContext *context, const ChatCompletionRequest *request,
                                ServerWriter<ChatCompletionStreamResponse> *writer) override
    {
        json req_json;
        if (request->max_tokens() > 0) req_json["max_tokens"] = request->max_tokens();
        if (!request->model().empty()) req_json["model"] = request->model();

        if (request->messages_size() > 0)
        {
            json messages = json::array();
            for (int i = 0; i < request->messages_size(); ++i)
            {
                auto &msg = request->messages(i);
                messages.push_back({{"role", msg.role()}, {"content", msg.content()}});
            }
            req_json["messages"] = messages;
        }
        else if (!request->prompt().empty())
        {
            req_json["prompt"] = request->prompt();
        }

        std::string req_body = req_json.dump();
        auto        session = std::make_shared<Session>();

        try
        {
            handler_(req_body, true, session);
        }
        catch (const std::exception &e)
        {
            return Status(grpc::StatusCode::INTERNAL, "Internal Server Error: " + std::string(e.what()));
        }

        bool is_thinking = false;

        // Consumer loop runs on the gRPC thread, unpacking chunks and handling network writes
        while (true)
        {
            std::pair<std::string, bool> item;
            {
                std::unique_lock<std::mutex> lock(session->mtx);
                session->cv.wait(lock, [&]() { return !session->output_queue.empty() || session->is_finished; });

                if (session->output_queue.empty() && session->is_finished) break;

                item = session->output_queue.front();
                session->output_queue.pop();
            }

            std::string &chunk_str = item.first;
            bool         is_finished = item.second;

            if (!chunk_str.empty() || is_finished)
            {
                std::string output_text = chunk_str;

                if (output_text == "<think>")
                {
                    is_thinking = true;
                    output_text = "";
                }
                else if (output_text == "</think>")
                {
                    is_thinking = false;
                    output_text = "";
                }
                else if (output_text.find("<|im_end|>") != std::string::npos ||
                         output_text.find("<|endoftext|>") != std::string::npos)
                {
                    output_text = "";
                }

                if (!output_text.empty() || is_finished)
                {
                    ChatCompletionStreamResponse resp;
                    resp.set_id("chatcmpl-firefly");
                    resp.set_object("chat.completion.chunk");
                    resp.set_created(std::time(nullptr));
                    resp.set_model(req_json.value("model", "qwen3"));

                    auto *choice = resp.add_choices();
                    choice->set_index(0);

                    if (is_finished)
                    {
                        choice->set_finish_reason("stop");
                    }
                    else
                    {
                        auto *delta = choice->mutable_delta();
                        if (is_thinking)
                        {
                            delta->set_reasoning_content(scrub_utf8(output_text));
                        }
                        else
                        {
                            delta->set_content(scrub_utf8(output_text));
                        }
                    }

                    writer->Write(resp);
                }
            }

            if (is_finished) break;
        }

        return Status::OK;
    }

private:
    std::function<void(const std::string &, bool, std::shared_ptr<Session>)> handler_;
};

bool GrpcAdapter::start_server(int                                                                      port,
                               std::function<void(const std::string &, bool, std::shared_ptr<Session>)> handler)
{
    std::string          server_address("0.0.0.0:" + std::to_string(port));
    InferenceServiceImpl service(handler);

    grpc::EnableDefaultHealthCheckService(true);
    grpc::reflection::InitProtoReflectionServerBuilderPlugin();

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Starting gRPC server on " << server_address << "..." << std::endl;
    server->Wait();
    return true;
}

}  // namespace server
}  // namespace firefly
