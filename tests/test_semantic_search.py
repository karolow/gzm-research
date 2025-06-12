import faiss
import numpy as np
import pytest

# Import the full module for patching
import research.llm.semantic_search as ss
from research.llm.semantic_search import semantic_rerank, semantic_search


@pytest.fixture(autouse=True)
def setup_fake(monkeypatch):
    # Setup for dict format tests
    dict_examples = [
        {"inputs": "apple"},
        {"inputs": "banana"},
        {"inputs": "orange"},
    ]

    # Setup for string format tests
    string_examples = [
        "apple",
        "banana",
        "orange",
    ]

    # Choose which format to use for tests - default to dict format
    examples = dict_examples

    # Monkey-patch examples and embed_text
    monkeypatch.setattr(ss, "_examples", examples)

    def fake_embed(text: str) -> np.ndarray:
        mapping = {
            "apple": [1.0, 0.0, 0.0],
            "banana": [0.0, 1.0, 0.0],
            "orange": [0.0, 0.0, 1.0],
        }
        return np.array(mapping[text], dtype="float32")

    monkeypatch.setattr(ss, "embed_text", fake_embed)
    # Build FAISS index with normalized embeddings
    embs = np.vstack(
        [fake_embed(ex["inputs"] if isinstance(ex, dict) else ex) for ex in examples]
    )
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    monkeypatch.setattr(ss, "_index", index)


@pytest.fixture
def setup_string_examples(monkeypatch):
    # Setup for string format tests
    examples = [
        "apple",
        "banana",
        "orange",
    ]

    # Monkey-patch examples and embed_text
    monkeypatch.setattr(ss, "_examples", examples)

    def fake_embed(text: str) -> np.ndarray:
        mapping = {
            "apple": [1.0, 0.0, 0.0],
            "banana": [0.0, 1.0, 0.0],
            "orange": [0.0, 0.0, 1.0],
        }
        return np.array(mapping[text], dtype="float32")

    monkeypatch.setattr(ss, "embed_text", fake_embed)
    # Build FAISS index with normalized embeddings
    embs = np.vstack([fake_embed(ex) for ex in examples])
    faiss.normalize_L2(embs)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    monkeypatch.setattr(ss, "_index", index)


def test_semantic_search_excludes_self():
    idxs = semantic_search("apple", top_k=2)
    assert 0 not in idxs, "Should exclude the query itself"
    assert set(idxs) == {1, 2}


def test_semantic_search_with_exclude():
    idxs = semantic_search("banana", top_k=1, exclude_indices=[2])
    assert 1 not in idxs, "Should exclude banana itself"
    assert 2 not in idxs, "Should exclude the specified index 2"
    assert idxs == [0]


def test_semantic_search_with_string_examples(setup_string_examples):
    idxs = semantic_search("apple", top_k=2)
    assert 0 not in idxs, "Should exclude the query itself"
    assert set(idxs) == {1, 2}


class DummyReranker:
    def __init__(self, api_key=None):
        pass

    def rerank(self, query, docs, top_n):
        # return docs in reverse order for testing
        return [(d, float(len(d))) for d in reversed(docs)]


def test_semantic_rerank(monkeypatch):
    # Monkey-patch JinaReranker directly
    monkeypatch.setattr(ss, "JinaReranker", DummyReranker)
    cases = semantic_rerank("orange", top_n=2)
    assert isinstance(cases, list)
    assert len(cases) == 2
    # Inputs should match dummy order - DummyReranker reverses the order
    inputs_list = [c["inputs"] for c in cases]
    assert inputs_list == ["apple", "banana"]


def test_semantic_rerank_with_string_examples(setup_string_examples, monkeypatch):
    # Monkey-patch JinaReranker directly
    monkeypatch.setattr(ss, "JinaReranker", DummyReranker)
    cases = semantic_rerank("orange", top_n=2)
    assert isinstance(cases, list)
    assert len(cases) == 2
    # Should return string examples in the expected order
    assert cases == ["apple", "banana"]


# Skip the real index tests - these require additional setup and can be tested manually
#
# @pytest.mark.skipif(
#     not ss.INDEX_PATH.exists() or not ss.EXAMPLES_PATH.exists(),
#     reason="FAISS index or examples not built, skipping real index tests"
# )
# def test_real_semantic_search_self_exclusion():
#     # Force loading resources from the real files
#     monkeypatch_module = pytest.MonkeyPatch()
#     monkeypatch_module.setattr(ss, '_index', None)
#     monkeypatch_module.setattr(ss, '_examples', None)
#
#     # Load the actual resources
#     ss._load_resources()
#     examples = ss._examples
#     if not examples:
#         pytest.skip("No examples loaded")
#
#     # Get the first example as a query
#     query_text = ""
#     if isinstance(examples[0], dict):
#         query_text = (
#             examples[0].get("inputs", "") or
#             examples[0].get("question", "") or
#             examples[0].get("query", "") or
#             str(examples[0])
#         )
#     else:
#         query_text = str(examples[0])
#
#     # Override the fake_embed from autouse fixture with the real embed_text
#     real_embed = ss.embed_text
#     monkeypatch_module.setattr(ss, 'embed_text', real_embed)
#
#     # Find similar documents
#     idxs = ss.semantic_search(query_text, top_k=5)
#
#     # Should exclude the query's own index (0)
#     assert 0 not in idxs
#     assert len(idxs) > 0, "Expected at least one result"
#     assert all(isinstance(i, int) for i in idxs)
#     assert len(idxs) <= 5
#
#     # Clean up
#     monkeypatch_module.undo()
#
#
# @pytest.mark.skipif(
#     not ss.INDEX_PATH.exists() or not ss.EXAMPLES_PATH.exists(),
#     reason="FAISS index or examples not built, skipping real index tests"
# )
# def test_real_semantic_search_types_and_length():
#     # Create a monkeypatch that won't interfere with the autouse fixture
#     monkeypatch_module = pytest.MonkeyPatch()
#     monkeypatch_module.setattr(ss, '_index', None)
#     monkeypatch_module.setattr(ss, '_examples', None)
#
#     # Load the actual resources
#     ss._load_resources()
#
#     # Override the fake_embed from autouse fixture with the real embed_text
#     real_embed = ss.embed_text
#     monkeypatch_module.setattr(ss, 'embed_text', real_embed)
#
#     # Query a random text, ensure valid index list
#     idxs = ss.semantic_search("how many people responded to the survey", top_k=3)
#     assert isinstance(idxs, list)
#     assert all(isinstance(i, int) for i in idxs)
#     assert len(idxs) <= 3
#
#     # Clean up
#     monkeypatch_module.undo()
