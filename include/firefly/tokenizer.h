#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace firefly
{

class Tokenizer
{
public:
    Tokenizer() = default;

    /**
     * @brief Loads from huggingface tokenizer.json.
     *
     * @param path The path to the tokenizer.json file.
     * @return bool True if successful, false otherwise.
     */
    bool load(const std::string& path);

    /**
     * @brief Encodes a string to a sequence of token IDs.
     *
     * @param text The input string to encode.
     * @return std::vector<int> Resulting token IDs.
     */
    std::vector<int> encode(const std::string& text) const;

    /**
     * @brief Decodes a single token ID to a string.
     *
     * @param id The token ID.
     * @return std::string Decoded string.
     */
    std::string decode(int id) const;

    /**
     * @brief Decodes a sequence of token IDs to a string.
     *
     * @param ids The sequence of token IDs.
     * @return std::string Decoded string.
     */
    std::string decode(const std::vector<int>& ids) const;

    int vocab_size() const { return id_to_token_.size(); }

private:
    std::unordered_map<int, std::string> id_to_token_;
    std::unordered_map<std::string, int> token_to_id_;

    /** @brief BPE Merges */
    std::unordered_map<std::string, int> merge_ranks_;

    /** @brief Precomputed byte-to-char mapping for byte-level BPE */
    std::unordered_map<unsigned char, std::string> byte_encoder_;
    std::unordered_map<std::string, unsigned char> byte_decoder_;

    void init_byte_encoder();
};

}  // namespace firefly
