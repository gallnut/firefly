#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <unicode/regex.h>

#include "firefly/core/error.h"

namespace firefly::model
{

/** @brief Hugging Face-compatible byte-level BPE tokenizer with special-token support. */
class Tokenizer
{
public:
    /** @brief Constructs an empty tokenizer that must be loaded before use. */
    Tokenizer() = default;
    /** @brief Releases ICU regular-expression resources and vocabulary storage. */
    ~Tokenizer();

    /**
     * @brief Loads from huggingface tokenizer.json.
     *
     * @param path The path to the tokenizer.json file.
     * @return Success or a structured I/O, JSON, vocabulary, or ICU error.
     */
    Status load(const std::string& path);

    /**
     * @brief Encodes a string to a sequence of token IDs.
     *
     * @param text The input string to encode.
     * @return Resulting token IDs or a structured tokenizer error.
     */
    Result<std::vector<int>> encode(const std::string& text) const;

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

    /**
     * @brief Returns the number of token IDs loaded from the vocabulary.
     * @return Vocabulary entry count, or zero before a tokenizer is loaded.
     */
    int  vocab_size() const { return id_to_token_.size(); }
    /**
     * @brief Returns the numeric ID for an exact token string.
     * @param token Vocabulary token in its stored encoded representation.
     * @return Token ID, or `-1` when no exact entry exists.
     */
    int  token_id(const std::string& token) const;
    /**
     * @brief Returns whether an exact token string exists in the loaded vocabulary.
     * @param token Vocabulary token in its stored encoded representation.
     * @return `true` when `token_to_id_` contains the supplied key.
     */
    bool has_token(const std::string& token) const;

private:
    std::unordered_map<int, std::string> id_to_token_; ///< Token text indexed by vocabulary ID.
    std::unordered_map<std::string, int> token_to_id_; ///< Vocabulary ID indexed by exact token text.
    std::vector<std::string>             special_tokens_; ///< Special tokens ordered for boundary matching.

    std::unordered_map<std::string, int> merge_ranks_; ///< Pair-string to byte-pair merge priority.

    std::unordered_map<unsigned char, std::string> byte_encoder_; ///< Reversible byte-to-Unicode encoding table.
    std::unordered_map<std::string, unsigned char> byte_decoder_; ///< Reverse Unicode-symbol-to-byte table.
    std::shared_ptr<const icu::RegexPattern> pretokenizer_pattern_; ///< Compiled tokenizer segmentation pattern.

    /** @brief Initializes reversible GPT-style byte-to-Unicode lookup tables. */
    void init_byte_encoder();
    /**
     * @brief Applies byte encoding and BPE merges to one pre-tokenized segment.
     * @param text Segment selected by the ICU pre-tokenizer.
     * @param ids Destination vector to which produced vocabulary IDs are appended.
     */
    Status encode_pretokenized_text(const std::string& text, std::vector<int>& ids) const;
    /**
     * @brief Splits ordinary text with the configured ICU pattern and appends token IDs.
     * @param text UTF-8 substring that contains no recognized special token.
     * @param ids Destination vector receiving encoded IDs.
     */
    Status encode_normal_text(const std::string& text, std::vector<int>& ids) const;
    /**
     * @brief Finds a special token beginning at a byte offset.
     * @param text Full UTF-8 source string.
     * @param pos Candidate byte offset.
     * @param token Receives the matched token text.
     * @param id Receives the matched vocabulary ID.
     * @return `true` when a configured special token begins exactly at `pos`.
     */
    bool match_special_token(const std::string& text, size_t pos, std::string& token, int& id) const;
};

}  // namespace firefly::model
