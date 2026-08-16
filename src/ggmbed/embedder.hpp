#pragma once

#include <llama.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ggmbed {

struct EmbeddingBatchResult {
  std::vector<float> embeddings; // contiguous row-major (batch_size * dim)
  size_t batch_size = 0;
  size_t dim = 0;
};

class Embedder {
public:
  Embedder(const std::string &gguf_path, const std::string &tokenizer_path,
           bool do_lower_case, int num_threads, int cls_token_id,
           int sep_token_id, int pooling_type);

  ~Embedder();

  // Disable copying to safeguard raw llama pointers
  Embedder(const Embedder &) = delete;
  Embedder &operator=(const Embedder &) = delete;

  // Move semantics
  Embedder(Embedder &&other) noexcept;
  Embedder &operator=(Embedder &&other) noexcept;

  EmbeddingBatchResult
  encode_tokens(const std::vector<std::vector<int32_t>> &batch_seqs,
                bool normalize);

  EmbeddingBatchResult encode_texts(const std::vector<std::string> &texts,
                                   size_t max_length, bool normalize);

  int get_embedding_dim() const { return model_embd_dim_; }
  int get_cls_token_id() const { return cls_token_id_; }
  int get_sep_token_id() const { return sep_token_id_; }
  int get_pooling_type() const { return pooling_type_; }

private:
  struct llama_model *llama_model_ = nullptr;
  struct llama_context *llama_ctx_ = nullptr;

  bool do_lower_case_ = false;
  int cls_token_id_ = -1;
  int sep_token_id_ = -1;
  int pooling_type_ = 0;
  int model_embd_dim_ = 0;
};

} // namespace ggmbed
