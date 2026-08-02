#include "firefly/execution/engine.h"

#include <algorithm>

namespace firefly::execution
{

void Engine::process_outputs(const std::vector<scheduler::SequencePtr>& sequences)
{
    for (const auto& request : sequences)
    {
        if (request->generated_tokens.empty()) continue;

        int next_token_id = request->generated_tokens.back();
        std::string text = tokenizer_.decode(next_token_id);
        bool hit_stop_token =
            (tokenizer_.has_token("<|im_end|>") && next_token_id == tokenizer_.token_id("<|im_end|>")) ||
            (tokenizer_.has_token("<|endoftext|>") && next_token_id == tokenizer_.token_id("<|endoftext|>"));
        bool is_finished = hit_stop_token || request->generated_tokens.size() >= static_cast<size_t>(request->max_tokens);

        request->utf8_buffer += text;
        size_t valid_length = 0;
        size_t length = request->utf8_buffer.length();
        for (size_t width = 1; width <= std::min<size_t>(4, length); ++width)
        {
            unsigned char byte = request->utf8_buffer[length - width];
            if ((byte & 0x80) == 0)
            {
                valid_length = length;
                break;
            }
            if ((byte & 0xC0) == 0x80) continue;
            if ((byte & 0xE0) == 0xC0)
            {
                valid_length = width >= 2 ? length : length - width;
                break;
            }
            if ((byte & 0xF0) == 0xE0)
            {
                valid_length = width >= 3 ? length : length - width;
                break;
            }
            if ((byte & 0xF8) == 0xF0)
            {
                valid_length = width >= 4 ? length : length - width;
                break;
            }
            valid_length = length;
            break;
        }
        if (valid_length == 0 && length >= 4) valid_length = length;
        if (is_finished) valid_length = length;

        std::string valid_text = request->utf8_buffer.substr(0, valid_length);
        request->utf8_buffer.erase(0, valid_length);
        if (result_queue_ && (!valid_text.empty() || is_finished))
        {
            if (is_finished)
                result_queue_->push(request->id, valid_text, true, static_cast<int>(request->prompt_tokens.size()),
                                    static_cast<int>(request->generated_tokens.size()));
            else
                result_queue_->push(request->id, valid_text, false);
        }

        if (is_finished) scheduler_.finish_sequence(request);
    }
}

}  // namespace firefly::execution
