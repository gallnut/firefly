#include "firefly/execution/engine.h"

#include <algorithm>

#include "firefly/core/logging.h"

namespace firefly::execution
{

void Engine::process_outputs(const std::vector<scheduler::SequencePtr>& sequences)
{
    for (const auto& request : sequences)
    {
        if (request->generated_tokens.empty()) continue;

        std::string pending_text;
        bool        is_finished = false;
        bool        hit_stop_token = false;
        while (request->emitted_tokens < request->generated_tokens.size())
        {
            const size_t token_index = request->emitted_tokens++;
            const int    next_token_id = request->generated_tokens[token_index];
            FIREFLY_LOG_TRACE("generation", "request_id={} token_index={} token_id={}", request->id, token_index,
                              next_token_id);
            pending_text += tokenizer_.decode(next_token_id);
            hit_stop_token = !request->ignore_eos &&
                ((tokenizer_.has_token("<|im_end|>") && next_token_id == tokenizer_.token_id("<|im_end|>")) ||
                 (tokenizer_.has_token("<|endoftext|>") && next_token_id == tokenizer_.token_id("<|endoftext|")));
            if (hit_stop_token) request->generated_tokens.resize(token_index + 1);
            is_finished = hit_stop_token ||
                          (token_index + 1 == request->generated_tokens.size() &&
                           request->generated_tokens.size() >= static_cast<size_t>(request->max_tokens));
            if (is_finished) break;
        }
        if (request->emitted_tokens == 0) continue;

        FIREFLY_LOG_DEBUG("generation", "request_id={} emitted={} generated={} finished={} stop_token={}", request->id,
                          request->emitted_tokens, request->generated_tokens.size(), is_finished, hit_stop_token);

        request->utf8_buffer += pending_text;
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

        if (is_finished)
        {
            if (speculative_decoder_ != nullptr) speculative_decoder_->erase(request->id);
            scheduler_.finish_sequence(request);
        }
    }
}

}  // namespace firefly::execution
