#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "embedder.hpp"

#include <cstring>

namespace nb = nanobind;

static nb::ndarray<float, nb::numpy, nb::device::cpu>
to_ndarray(const ggmbed::EmbeddingBatchResult &res) {
  if (res.batch_size == 0) {
    float *empty_data = new float[0];
    size_t shape[2] = {0, res.dim};
    nb::capsule owner(empty_data, [](void *p) noexcept {
      delete[] static_cast<float *>(p);
    });
    return nb::ndarray<float, nb::numpy, nb::device::cpu>(empty_data, 2, shape,
                                                          owner);
  }

  size_t total_elements = res.batch_size * res.dim;
  float *result_data = new float[total_elements];
  std::memcpy(result_data, res.embeddings.data(),
              total_elements * sizeof(float));

  size_t shape[2] = {res.batch_size, res.dim};
  nb::capsule owner(result_data, [](void *p) noexcept {
    delete[] static_cast<float *>(p);
  });

  return nb::ndarray<float, nb::numpy, nb::device::cpu>(result_data, 2, shape,
                                                        owner);
}

NB_MODULE(_core, m) {
  m.doc() = "ggmbed native C++ dense embedding core (GGUF-only)";

  nb::class_<ggmbed::Embedder>(m, "Embedder")
      .def(nb::init<const std::string &, const std::string &, bool, int, int,
                    int, int>(),
           nb::arg("gguf_path"), nb::arg("tokenizer_path"),
           nb::arg("do_lower_case"), nb::arg("num_threads"),
           nb::arg("cls_token_id"), nb::arg("sep_token_id"),
           nb::arg("pooling_type"))
      .def(
          "encode",
          [](ggmbed::Embedder &self, const std::vector<std::string> &texts,
             size_t max_length, bool normalize) {
            ggmbed::EmbeddingBatchResult res;
            {
              nb::gil_scoped_release release;
              res = self.encode_texts(texts, max_length, normalize);
            }
            return to_ndarray(res);
          },
          nb::arg("texts"), nb::arg("max_length"), nb::arg("normalize"))
      .def(
          "encode_tokens",
          [](ggmbed::Embedder &self,
             const std::vector<std::vector<int32_t>> &batch_seqs,
             bool normalize) {
            ggmbed::EmbeddingBatchResult res;
            {
              nb::gil_scoped_release release;
              res = self.encode_tokens(batch_seqs, normalize);
            }
            return to_ndarray(res);
          },
          nb::arg("batch_seqs"), nb::arg("normalize"))
      .def_prop_ro("model_embd_dim_", &ggmbed::Embedder::get_embedding_dim)
      .def_prop_ro("cls_token_id", &ggmbed::Embedder::get_cls_token_id)
      .def_prop_ro("sep_token_id", &ggmbed::Embedder::get_sep_token_id)
      .def_prop_ro("pooling_type", &ggmbed::Embedder::get_pooling_type);
}
