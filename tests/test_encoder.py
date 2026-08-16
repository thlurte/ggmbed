import os
import tempfile
import numpy as np
from unittest.mock import MagicMock, patch
import pytest
from ggmbed import Embedder

@pytest.fixture
def mock_dependencies():
    temp_dir = tempfile.TemporaryDirectory()
    model_path = os.path.join(temp_dir.name, "model.gguf")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(model_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"model": {"vocab": {}}}')
        
    yield model_path, tokenizer_path
    
    temp_dir.cleanup()

@patch("ggmbed.encoder.CppEmbedder")
def test_encoder_init_and_encode(mock_cpp_encoder_cls, mock_dependencies):
    model_path, tokenizer_path = mock_dependencies
    
    mock_cpp_encoder = MagicMock()
    mock_cpp_encoder_cls.return_value = mock_cpp_encoder
    
    dummy_embs = np.array([[1.0, 0.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    
    mock_cpp_encoder.encode.return_value = dummy_embs
    
    encoder = Embedder(model_path, tokenizer_path)
    embs = encoder.encode(["test query 1", "test query 2"], max_length=128, normalize=True)
    mock_cpp_encoder.encode.assert_called_with(["test query 1", "test query 2"], 128, True)
    assert np.array_equal(embs, dummy_embs)

@patch("ggmbed.encoder.CppEmbedder")
def test_encoder_init_with_directory(mock_cpp_encoder_cls):
    temp_dir = tempfile.TemporaryDirectory()
    gguf_path = os.path.join(temp_dir.name, "model.gguf")
    tokenizer_path = os.path.join(temp_dir.name, "tokenizer.json")
    
    with open(gguf_path, "wb") as f:
        f.write(b"mock_model_data")
    with open(tokenizer_path, "w") as f:
        f.write('{"model": {"vocab": {}}}')
        
    encoder = Embedder(temp_dir.name)
    
    mock_cpp_encoder_cls.assert_called_with(
        gguf_path,
        tokenizer_path,
        False,          # do_lower_case
        0,              # num_threads
        101,            # cls_token_id
        102,            # sep_token_id
        0               # pooling_type (mean)
    )
    
    temp_dir.cleanup()

def test_real_embedding_end_to_end():
    print("\nRunning real end-to-end embedding test with all-MiniLM-L6-v2...")
    encoder = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    
    texts = [
        "The cat is sleeping on the couch",
        "A kitten is napping on the sofa",
        "General relativity explains gravitational phenomena in physics"
    ]
    embs = encoder.encode(texts, max_length=128, normalize=True)
    
    assert embs.shape == (3, 384)
    for i in range(3):
        norm = np.linalg.norm(embs[i])
        assert np.allclose(norm, 1.0, atol=1e-4)
    
    sim_cat_kitten = np.dot(embs[0], embs[1])
    sim_cat_physics = np.dot(embs[0], embs[2])
    
    print(f"Similarity(cat, kitten): {sim_cat_kitten:.4f}")
    print(f"Similarity(cat, physics): {sim_cat_physics:.4f}")
    
    assert sim_cat_kitten > 0.70
    assert sim_cat_physics < 0.10
    assert sim_cat_kitten > sim_cat_physics + 0.50

def test_bge_embedding_end_to_end():
    print("\nRunning real BGE embedding test...")
    encoder = Embedder("BAAI/bge-small-en-v1.5", quantization="Q8_0")
    texts = ["hello world", "bge model uses cls pooling"]
    embs = encoder.encode(texts, max_length=128, normalize=True)
    
    assert embs.shape == (2, 384)
    norm = np.linalg.norm(embs[0])
    assert np.allclose(norm, 1.0, atol=1e-5)
    assert not np.allclose(embs[0], 0.0)
    assert not np.allclose(embs[0], embs[1])

def test_denseon_embedding_end_to_end():
    print("\nRunning real DenseOn embedding test...")
    encoder = Embedder("lightonai/DenseOn", quantization="Q8_0")
    texts = ["hello world", "denseon model uses cls pooling and has 768 dimensions"]
    embs = encoder.encode(texts, max_length=128, normalize=True)
    
    assert embs.shape == (2, 768)
    norm = np.linalg.norm(embs[0])
    assert np.allclose(norm, 1.0, atol=1e-5)
    assert not np.allclose(embs[0], 0.0)
    assert not np.allclose(embs[0], embs[1])

def test_pytorch_numerical_equivalence():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        pytest.skip("sentence_transformers not installed")
        
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gg_model = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Quantum mechanics is a fundamental branch of physics.",
        "Deep neural networks learn distributed representations."
    ]
    st_embs = st_model.encode(sentences, normalize_embeddings=True)
    gg_embs = gg_model.encode(sentences, normalize=True)
    
    for i in range(len(sentences)):
        cosine_sim = np.dot(gg_embs[i], st_embs[i])
        print(f"Sentence {i+1} alignment with PyTorch: {cosine_sim:.5f}")
        assert cosine_sim > 0.999, f"Sentence {i+1} alignment {cosine_sim:.5f} is below 0.999"
