#include "embedder.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace ggmbed {

Embedder::Embedder(const std::string &gguf_path,
                   const std::string &tokenizer_path, bool do_lower_case,
                   int num_threads, int cls_token_id, int sep_token_id,
                   int pooling_type)
    : do_lower_case_(do_lower_case), cls_token_id_(cls_token_id),
      sep_token_id_(sep_token_id), pooling_type_(pooling_type) {

  llama_backend_init();

  llama_model_params model_params = llama_model_default_params();
  llama_model_ = llama_model_load_from_file(gguf_path.c_str(), model_params);
  if (!llama_model_) {
    throw std::runtime_error("Failed to load GGUF model: " + gguf_path);
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);
  if (cls_token_id_ <= 0) {
    cls_token_id_ = llama_vocab_bos(vocab);
    if (cls_token_id_ < 0) {
      cls_token_id_ = 101;
    }
  }
  if (sep_token_id_ <= 0) {
    sep_token_id_ = llama_vocab_sep(vocab);
    if (sep_token_id_ < 0) {
      sep_token_id_ = llama_vocab_eos(vocab);
    }
    if (sep_token_id_ < 0) {
      sep_token_id_ = 102;
    }
  }

  llama_context_params ctx_params = llama_context_default_params();
  ctx_params.embeddings = true;
  int actual_threads = num_threads;
  if (actual_threads <= 0) {
    actual_threads =
        std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
  }
  ctx_params.n_threads = actual_threads;
  ctx_params.n_threads_batch = actual_threads;
  ctx_params.n_ctx = 512;
  ctx_params.n_batch = 512;
  ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

  if (pooling_type_ == 1) {
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_CLS;
  } else {
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
  }

  llama_ctx_ = llama_init_from_model(llama_model_, ctx_params);
  if (!llama_ctx_) {
    throw std::runtime_error("Failed to create llama.cpp context");
  }

  model_embd_dim_ = llama_model_n_embd(llama_model_);
}

Embedder::~Embedder() {
  if (llama_ctx_) {
    llama_free(llama_ctx_);
    llama_ctx_ = nullptr;
  }
  if (llama_model_) {
    llama_model_free(llama_model_);
    llama_model_ = nullptr;
  }
}

Embedder::Embedder(Embedder &&other) noexcept
    : llama_model_(other.llama_model_), llama_ctx_(other.llama_ctx_),
      do_lower_case_(other.do_lower_case_),
      cls_token_id_(other.cls_token_id_),
      sep_token_id_(other.sep_token_id_),
      pooling_type_(other.pooling_type_),
      model_embd_dim_(other.model_embd_dim_) {
  other.llama_model_ = nullptr;
  other.llama_ctx_ = nullptr;
}

Embedder &Embedder::operator=(Embedder &&other) noexcept {
  if (this != &other) {
    if (llama_ctx_) {
      llama_free(llama_ctx_);
    }
    if (llama_model_) {
      llama_model_free(llama_model_);
    }
    llama_model_ = other.llama_model_;
    llama_ctx_ = other.llama_ctx_;
    do_lower_case_ = other.do_lower_case_;
    cls_token_id_ = other.cls_token_id_;
    sep_token_id_ = other.sep_token_id_;
    pooling_type_ = other.pooling_type_;
    model_embd_dim_ = other.model_embd_dim_;

    other.llama_model_ = nullptr;
    other.llama_ctx_ = nullptr;
  }
  return *this;
}

EmbeddingBatchResult
Embedder::encode_tokens(const std::vector<std::vector<int32_t>> &batch_seqs,
                        bool normalize) {
  EmbeddingBatchResult result;
  if (batch_seqs.empty()) {
    result.batch_size = 0;
    result.dim = model_embd_dim_;
    return result;
  }

  size_t batch_size = batch_seqs.size();
  size_t output_dim = model_embd_dim_;
  result.batch_size = batch_size;
  result.dim = output_dim;
  result.embeddings.resize(batch_size * output_dim, 0.0f);

  for (size_t b = 0; b < batch_size; ++b) {
    const std::vector<int32_t> &seq = batch_seqs[b];
    size_t decode_len = seq.size();
    if (decode_len == 0) {
      continue;
    }

    llama_batch batch = llama_batch_init(decode_len, 0, 1);
    batch.n_tokens = decode_len;

    for (size_t s = 0; s < decode_len; ++s) {
      batch.token[s] = seq[s];
      batch.pos[s] = s;
      batch.n_seq_id[s] = 1;
      batch.seq_id[s][0] = 0;
      batch.logits[s] = true;
    }

    int res = llama_encode(llama_ctx_, batch);
    if (res != 0) {
      res = llama_decode(llama_ctx_, batch);
    }

    if (res != 0) {
      llama_batch_free(batch);
      throw std::runtime_error("llama_encode failed with error code: " +
                               std::to_string(res));
    }

    float *out_vec = result.embeddings.data() + b * output_dim;
    const float *embd = llama_get_embeddings_seq(llama_ctx_, 0);
    if (!embd) {
      embd = llama_get_embeddings_ith(llama_ctx_, (pooling_type_ == 1) ? 0 : -1);
    }

    if (!embd) {
      llama_batch_free(batch);
      throw std::runtime_error("Failed to retrieve embeddings from llama context");
    }

    std::memcpy(out_vec, embd, output_dim * sizeof(float));

    if (normalize) {
      float norm_sq = 0.0f;
      for (size_t k = 0; k < output_dim; ++k) {
        norm_sq += out_vec[k] * out_vec[k];
      }
      float norm = std::sqrt(norm_sq);
      float norm_scale = norm > 1e-12f ? 1.0f / norm : 0.0f;
      for (size_t k = 0; k < output_dim; ++k) {
        out_vec[k] *= norm_scale;
      }
    }

    llama_batch_free(batch);
  }

  return result;
}

EmbeddingBatchResult
Embedder::encode_texts(const std::vector<std::string> &texts,
                       size_t max_length, bool normalize) {
  if (texts.empty()) {
    EmbeddingBatchResult result;
    result.batch_size = 0;
    result.dim = model_embd_dim_;
    return result;
  }

  const struct llama_vocab *vocab = llama_model_get_vocab(llama_model_);
  std::vector<std::vector<int32_t>> batch_seqs(texts.size());

  for (size_t b = 0; b < texts.size(); ++b) {
    const std::string *text_ptr = &texts[b];
    std::string lower_buf;
    if (do_lower_case_) {
      lower_buf = texts[b];
      std::transform(lower_buf.begin(), lower_buf.end(), lower_buf.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      text_ptr = &lower_buf;
    }

    std::vector<llama_token> raw_ids(text_ptr->length() * 4 + 32);
    int32_t n_tokens =
        llama_tokenize(vocab, text_ptr->c_str(), text_ptr->length(),
                       raw_ids.data(), raw_ids.size(), false, true);
    if (n_tokens < 0) {
      raw_ids.resize(-n_tokens);
      n_tokens =
          llama_tokenize(vocab, text_ptr->c_str(), text_ptr->length(),
                         raw_ids.data(), raw_ids.size(), false, true);
    }
    raw_ids.resize(std::max(0, n_tokens));

    std::vector<int32_t> seq;
    seq.push_back(cls_token_id_);
    for (llama_token id : raw_ids) {
      if (id != cls_token_id_ && id != sep_token_id_) {
        seq.push_back(id);
      }
    }
    if (seq.size() >= max_length) {
      seq.resize(max_length - 1);
    }
    seq.push_back(sep_token_id_);
    batch_seqs[b] = std::move(seq);
  }

  return encode_tokens(batch_seqs, normalize);
}

} // namespace ggmbed
