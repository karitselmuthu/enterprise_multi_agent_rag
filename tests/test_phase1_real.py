import tempfile
import unittest
from pathlib import Path

from phase0.phase1.dependency_graph import DependencyGraphBuilder, DependencyGraphStore
from phase0.phase1.ingestion import ASTChunker
from phase0.phase1.pipeline import Phase1IndexerPipeline
from phase0.phase1.qdrant import QdrantVectorStore


class _FakeNode:
    def __init__(self, node_type: str, start_byte: int, end_byte: int, children=None) -> None:
        self.type = node_type
        self.start_byte = start_byte
        self.end_byte = end_byte
        self.children = children or []


class _FakeTree:
    def __init__(self, root_node: _FakeNode) -> None:
        self.root_node = root_node


class _FakeParser:
    def __init__(self, root_node: _FakeNode) -> None:
        self._root_node = root_node

    def parse(self, _source: bytes) -> _FakeTree:
        return _FakeTree(self._root_node)


class _FakeQdrantClient:
    def __init__(self, **_kwargs) -> None:
        self.created = False
        self.upsert_calls = 0
        self.collection_name = ""
        self.points = []

    def collection_exists(self, _collection_name: str) -> bool:
        return self.created

    def create_collection(self, collection_name, **_kwargs) -> None:
        self.created = True
        self.collection_name = collection_name

    def upsert(self, collection_name, points, wait) -> None:
        self.collection_name = collection_name
        self.points.extend(points)
        self.upsert_calls += 1
        if wait is not True:
            raise AssertionError("Expected wait=True")

    def search(self, **_kwargs):
        return []


class Phase1RealTests(unittest.TestCase):
    def test_tree_sitter_chunking_uses_nodes(self) -> None:
        source = "import os\n\ndef hello():\n    return 1\n"
        import_node = _FakeNode("import_statement", 0, 9)
        fn_node = _FakeNode("function_definition", 11, len(source))
        root = _FakeNode("module", 0, len(source), [import_node, fn_node])
        chunker = ASTChunker(parser_resolver=lambda _language: _FakeParser(root))
        chunks = chunker.chunk(source, path="module.py", chunk_size=30)
        self.assertEqual(len(chunks), 2)
        self.assertIn("import os", chunks[0])
        self.assertIn("def hello", chunks[1])

    def test_dependency_graph_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.sqlite"
            store = DependencyGraphStore(db_path)
            store.upsert_node("core:app.py", "module", "core", "platform", {"path": "app.py"})
            edge_builder = DependencyGraphBuilder()
            content = "import os\nfrom pkg.mod import a\n"
            edges = edge_builder.extract_edges(
                repo="core",
                module_id="core:app.py",
                path="app.py",
                content=content,
            )
            for edge in edges:
                store.upsert_edge(edge)
            downstream = store.downstream("core:app.py")
            targets = {item.target for item in downstream}
            self.assertIn("os", targets)
            self.assertIn("pkg.mod", targets)

    def test_pipeline_indexes_qdrant_and_graph(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "graph.sqlite"
            graph_store = DependencyGraphStore(db_path)

            class _RestPointStruct:
                def __init__(self, id, vector, payload) -> None:
                    self.id = id
                    self.vector = vector
                    self.payload = payload

            class _RestDistance:
                COSINE = "Cosine"

            class _RestVectorParams:
                def __init__(self, size, distance) -> None:
                    self.size = size
                    self.distance = distance

            class _FakeStore(QdrantVectorStore):
                def ensure_collection(self, vector_size: int) -> None:
                    if self._client.collection_exists(self.collection_name):
                        return
                    self._client.create_collection(collection_name=self.collection_name)

                def upsert_chunks(self, chunks, dense_vectors) -> None:
                    points = []
                    for chunk, vector in zip(chunks, dense_vectors, strict=True):
                        points.append(
                            _RestPointStruct(
                                id=chunk.chunk_id,
                                vector=vector,
                                payload={"text": chunk.text, "repo": chunk.metadata.repo},
                            )
                        )
                    self._client.upsert(collection_name=self.collection_name, points=points, wait=True)

            vector_store = _FakeStore(
                url="http://localhost:6333",
                collection_name="code_chunks",
                client_factory=lambda **_kwargs: _FakeQdrantClient(),
            )
            source = "import requests\n\ndef f():\n    return 1\n"
            root = _FakeNode(
                "module",
                0,
                len(source),
                [
                    _FakeNode("import_statement", 0, 15),
                    _FakeNode("function_definition", 17, len(source)),
                ],
            )
            chunk_indexer = ASTChunker(parser_resolver=lambda _language: _FakeParser(root))

            from phase0.phase1.ingestion import ChunkIndexer

            pipeline = Phase1IndexerPipeline(
                chunk_indexer=ChunkIndexer(chunker=chunk_indexer),
                vector_store=vector_store,
                graph_store=graph_store,
            )
            result = pipeline.index_document(
                repo="core",
                path="svc.py",
                owner_team="platform",
                classification="internal",
                acl_roles=("dev",),
                content=source,
                embedding_fn=lambda text: [float(len(text)), 0.5, 1.0],
            )
            self.assertGreaterEqual(result.chunks_indexed, 1)
            self.assertGreaterEqual(result.dependencies_indexed, 1)
            self.assertEqual(vector_store._client.upsert_calls, 1)
            downstream = graph_store.downstream("core:svc.py")
            self.assertTrue(any(edge.target == "requests" for edge in downstream))


if __name__ == "__main__":
    unittest.main()
