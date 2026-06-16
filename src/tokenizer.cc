#include "firefly/tokenizer.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

using json = nlohmann::json;

namespace firefly
{

bool Tokenizer::load(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open())
    {
        std::cerr << "Failed to open tokenizer: " << path << std::endl;
        return false;
    }

    try
    {
        json j = json::parse(f);
        auto vocab = j["model"]["vocab"];

        init_byte_encoder();

        for (auto it = vocab.begin(); it != vocab.end(); ++it)
        {
            std::string token = it.key();
            int         id = it.value();

            id_to_token_[id] = token;
            token_to_id_[token] = id;
        }

        if (j.contains("added_tokens") && j["added_tokens"].is_array())
        {
            for (const auto& item : j["added_tokens"])
            {
                if (!item.contains("content") || !item.contains("id")) continue;

                std::string token = item["content"].get<std::string>();
                int         id = item["id"].get<int>();
                id_to_token_[id] = token;
                token_to_id_[token] = id;
                special_tokens_.push_back(token);
            }
        }

        std::sort(special_tokens_.begin(), special_tokens_.end(),
                  [](const auto& a, const auto& b) { return a.size() > b.size(); });

        if (j["model"].contains("merges"))
        {
            auto merges = j["model"]["merges"];
            int  rank = 0;
            for (const auto& merge : merges)
            {
                if (merge.is_string())
                {
                    std::string merge_str = merge.get<std::string>();
                    merge_ranks_[merge_str] = rank++;
                }
                else if (merge.is_array() && merge.size() == 2)
                {
                    std::string a = merge[0].get<std::string>();
                    std::string b = merge[1].get<std::string>();
                    merge_ranks_[a + " " + b] = rank++;
                }
            }
        }

        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error parsing tokenizer JSON: " << e.what() << std::endl;
        return false;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text) const
{
    std::vector<int> ids;
    if (text.empty()) return ids;

    size_t pos = 0;
    while (pos < text.size())
    {
        std::string special;
        int         special_id = -1;
        if (match_special_token(text, pos, special, special_id))
        {
            ids.push_back(special_id);
            pos += special.size();
            continue;
        }

        size_t next = pos + 1;
        while (next < text.size())
        {
            if (match_special_token(text, next, special, special_id)) break;
            ++next;
        }
        encode_pretokenized_text(text.substr(pos, next - pos), ids);
        pos = next;
    }

    return ids;
}

void Tokenizer::encode_pretokenized_text(const std::string& text, std::vector<int>& ids) const
{
    size_t pos = 0;
    while (pos < text.size())
    {
        unsigned char c = static_cast<unsigned char>(text[pos]);

        size_t start = pos;
        if (std::isspace(c))
        {
            while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
            {
                ++pos;
            }
            if (pos < text.size() && text[pos - 1] == ' ')
            {
                unsigned char next = static_cast<unsigned char>(text[pos]);
                if (!std::isspace(next))
                {
                    continue;
                }
            }
        }
        else
        {
            if (pos > 0 && text[pos - 1] == ' ')
            {
                --start;
            }

            if (std::isalpha(c))
            {
                ++pos;
                while (pos < text.size() && std::isalpha(static_cast<unsigned char>(text[pos])))
                {
                    ++pos;
                }
            }
            else if (std::isdigit(c))
            {
                ++pos;
                while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
                {
                    ++pos;
                }
            }
            else if ((c & 0x80) == 0)
            {
                ++pos;
                while (pos < text.size())
                {
                    unsigned char next = static_cast<unsigned char>(text[pos]);
                    if (std::isspace(next) || std::isalpha(next) || std::isdigit(next) || (next & 0x80) != 0)
                    {
                        break;
                    }
                    ++pos;
                }
            }
            else
            {
                ++pos;
                while (pos < text.size() && (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80)
                {
                    ++pos;
                }
            }
        }

        if (pos > start)
        {
            encode_normal_text(text.substr(start, pos - start), ids);
        }
    }
}

void Tokenizer::encode_normal_text(const std::string& text, std::vector<int>& ids) const
{
    if (text.empty()) return;

    // 1. Convert string to BPE bytes utilizing the byte encoder
    std::vector<std::string> bpe_chars;
    for (char c : text)
    {
        unsigned char byte = static_cast<unsigned char>(c);
        bpe_chars.push_back(byte_encoder_.at(byte));
    }

    // 2. Iteratively merge the best pairs
    while (bpe_chars.size() > 1)
    {
        int         best_rank = 1e9;
        int         best_idx = -1;
        std::string best_pair = "";

        for (size_t i = 0; i < bpe_chars.size() - 1; ++i)
        {
            std::string pair = bpe_chars[i] + " " + bpe_chars[i + 1];
            auto        it = merge_ranks_.find(pair);
            if (it != merge_ranks_.end())
            {
                if (it->second < best_rank)
                {
                    best_rank = it->second;
                    best_idx = i;
                    best_pair = bpe_chars[i] + bpe_chars[i + 1];
                }
            }
        }

        // If no more merges are possible, stop
        if (best_idx == -1)
        {
            break;
        }

        // Merge the best pair
        bpe_chars[best_idx] = best_pair;
        bpe_chars.erase(bpe_chars.begin() + best_idx + 1);
    }

    // 3. Map final BPE strings to token IDs
    for (const auto& token_str : bpe_chars)
    {
        auto it = token_to_id_.find(token_str);
        if (it != token_to_id_.end())
        {
            ids.push_back(it->second);
        }
        else
        {
            size_t i = 0;
            while (i < token_str.size())
            {
                unsigned char c = token_str[i];
                size_t        len = 1;
                if ((c & 0x80) == 0)
                    len = 1;
                else if ((c & 0xE0) == 0xC0)
                    len = 2;
                else if ((c & 0xF0) == 0xE0)
                    len = 3;
                else if ((c & 0xF8) == 0xF0)
                    len = 4;

                if (i + len > token_str.size()) len = token_str.size() - i;

                std::string piece = token_str.substr(i, len);
                auto        piece_it = token_to_id_.find(piece);
                if (piece_it == token_to_id_.end())
                {
                    throw std::runtime_error("Tokenizer cannot map BPE token: " + token_str);
                }
                ids.push_back(piece_it->second);
                i += len;
            }
        }
    }
}

std::string Tokenizer::decode(int id) const
{
    auto it = id_to_token_.find(id);
    if (it != id_to_token_.end())
    {
        std::string token = it->second;

        std::string decoded;
        size_t      i = 0;
        while (i < token.length())
        {
            unsigned char c = token[i];
            size_t        len = 1;
            if ((c & 0x80) == 0)
                len = 1;
            else if ((c & 0xE0) == 0xC0)
                len = 2;
            else if ((c & 0xF0) == 0xE0)
                len = 3;
            else if ((c & 0xF8) == 0xF0)
                len = 4;

            // Protect against malformed UTF-8 at the end of string
            if (i + len > token.length()) len = token.length() - i;

            std::string utf8_char = token.substr(i, len);
            auto        dec_it = byte_decoder_.find(utf8_char);
            if (dec_it != byte_decoder_.end())
            {
                decoded += (char)dec_it->second;
            }
            else
            {
                decoded += utf8_char;  // Fallback
            }
            i += len;
        }
        return decoded;
    }
    return "<unk>";
}

std::string Tokenizer::decode(const std::vector<int>& ids) const
{
    std::string result;
    for (int id : ids)
    {
        result += decode(id);
    }
    return result;
}

int Tokenizer::token_id(const std::string& token) const
{
    auto it = token_to_id_.find(token);
    if (it == token_to_id_.end())
    {
        throw std::runtime_error("Tokenizer missing token: " + token);
    }
    return it->second;
}

bool Tokenizer::has_token(const std::string& token) const { return token_to_id_.find(token) != token_to_id_.end(); }

bool Tokenizer::match_special_token(const std::string& text, size_t pos, std::string& token, int& id) const
{
    for (const auto& candidate : special_tokens_)
    {
        if (candidate.empty() || pos + candidate.size() > text.size()) continue;
        if (text.compare(pos, candidate.size(), candidate) == 0)
        {
            token = candidate;
            id = token_to_id_.at(candidate);
            return true;
        }
    }
    return false;
}

void Tokenizer::init_byte_encoder()
{
    // The standard GPT-2 / Qwen byte encoder maps bytes to printable unicode chars.
    // ASCII characters !-~ (33-126) and ¡-¬ (161-172), ®-ÿ (174-255) map to themselves.
    // The rest (0-32, 127-160, 173) are mapped to 256+.

    int n = 0;
    for (int b = 0; b < 256; ++b)
    {
        unsigned char byte = (unsigned char)b;
        std::string   mapped_char;

        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255))
        {
            // UTF8 encoding of the single byte (ASCII is 1 byte, 161+ is 2 bytes but in tokenizer mapped explicitly)
            // Actually, in standard BPE, the byte itself is just treated as the underlying character.
            // But we have to map it to a utf-8 string if it's > 127.
            if (b <= 127)
            {
                mapped_char = std::string(1, (char)b);
            }
            else
            {
                // Unicode codepoint b to UTF-8
                mapped_char += (char)(0xC0 | (b >> 6));
                mapped_char += (char)(0x80 | (b & 0x3F));
            }
        }
        else
        {
            // Map to 256 + n
            int codepoint = 256 + n;
            mapped_char += (char)(0xC0 | (codepoint >> 6));
            mapped_char += (char)(0x80 | (codepoint & 0x3F));
            n++;
        }

        byte_encoder_[byte] = mapped_char;
        byte_decoder_[mapped_char] = byte;
    }
}

}  // namespace firefly
