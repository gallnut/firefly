#pragma once

#include <memory>

namespace firefly
{
namespace model
{
class Tokenizer;
}
namespace execution
{
class Engine;
class ResultQueue;
}
namespace service
{

class Server
{
public:
    Server(execution::Engine& engine, const model::Tokenizer& tokenizer, execution::ResultQueue& results);
    ~Server();

    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;

    void run(int port);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace service
}  // namespace firefly
