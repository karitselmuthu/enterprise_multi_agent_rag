import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLATFORM_ROOT = REPO_ROOT / "code-rag-platform" / "src"
sys.path.insert(0, str(PLATFORM_ROOT))

from code_rag_platform.agents.orchestrator import OrchestratorState, PhaseOrchestrator, build_langgraph_state_machine
from code_rag_platform.agents.router import ExecutionRouter, RouteInput
from code_rag_platform.config.settings import AppSettings, ProviderSettings
from code_rag_platform.core.guardrails.dlp_scrubber import DLPScrubber
from code_rag_platform.core.ingestion.chunker import TreeSitterChunker
from code_rag_platform.core.retrieval.live_indexed_retriever import LiveIndexedRetriever


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
    def __init__(self, root: _FakeNode) -> None:
        self.root = root

    def parse(self, _payload: bytes) -> _FakeTree:
        return _FakeTree(self.root)


class _ChunkerForTest(TreeSitterChunker):
    def __init__(self, parser: _FakeParser) -> None:
        super().__init__()
        self._parser = parser

    def _get_parser(self, language: str):  # type: ignore[override]
        _ = language
        return self._parser


class _FakeRetriever:
    def __init__(self) -> None:
        self.index_calls = 0
        self.retrieve_calls = 0

    def index_document(self, repo, path, owner_team, classification, acl_roles, content):
        _ = (repo, path, owner_team, classification, acl_roles, content)
        self.index_calls += 1
        return (2, 1)

    def retrieve(self, query, repo, module_path, limit=5):
        _ = (query, repo, module_path, limit)
        self.retrieve_calls += 1
        return type(
            "Result",
            (),
            {
                "hits": [{"chunk_id": "c1", "score": 0.99, "payload": {"text": "owner=platform"}}],
                "dependencies": ["requests", "os"],
            },
        )()


class _FakeGraphStore:
    def downstream(self, _module_id):
        return [type("Edge", (), {"target": "dep.alpha"})(), type("Edge", (), {"target": "dep.beta"})()]


class _FakeVectorStore:
    def search(self, query_vector, limit=5):
        _ = (query_vector, limit)
        return [type("Hit", (), {"chunk_id": "chunk-1", "score": 0.88, "payload": {"text": "hello"}})()]


class _FakePipeline:
    def __init__(self) -> None:
        self.vector_store = _FakeVectorStore()
        self.indexed = 0

    def index_document(self, **_kwargs):
        self.indexed += 1
        return type("Indexed", (), {"chunks_indexed": 3, "dependencies_indexed": 2})()


class CodeRagPlatformTests(unittest.TestCase):
    def test_dlp_scrubber_masks_secret(self) -> None:
        scrubber = DLPScrubber()
        text = 'token="abcd1234" and user test@example.com'
        cleaned = scrubber.scrub(text)
        self.assertIn("[REDACTED_SECRET]", cleaned)
        self.assertIn("[REDACTED_PII]", cleaned)

    def test_execution_router_deep_escalates(self) -> None:
        router = ExecutionRouter()
        decision = router.classify(
            RouteInput(
                task_type="security_review",
                risk_level="high",
                confidence=0.7,
                evidence_coverage=0.5,
            )
        )
        self.assertEqual(decision.execution_path, "deep")
        self.assertTrue(decision.escalate_to_cloud)
        self.assertTrue(decision.needs_verification)

    def test_orchestrator_scrubs_high_risk_query(self) -> None:
        orchestrator = PhaseOrchestrator()
        state = orchestrator.run(
            OrchestratorState(
                user_query='api_key="secret" incident for checkout',
                task_type="incident_rca",
                risk_level="high",
                confidence=0.6,
                evidence_coverage=0.5,
            )
        )
        self.assertIn("[REDACTED_SECRET]", state.sanitized_query)
        self.assertTrue(state.route["escalate_to_cloud"])

    def test_orchestrator_uses_live_indexed_retrieval(self) -> None:
        retriever = _FakeRetriever()
        orchestrator = PhaseOrchestrator(retriever=retriever)
        state = orchestrator.run(
            OrchestratorState(
                user_query="who owns module",
                task_type="ownership_lookup",
                source_path="svc.py",
                source_content="import os\n\ndef f():\n  return 1\n",
            )
        )
        self.assertEqual(retriever.index_calls, 1)
        self.assertEqual(retriever.retrieve_calls, 1)
        self.assertEqual(len(state.retrieval_hits), 1)
        self.assertEqual(len(state.dependency_context), 2)
        self.assertIn("index_stats", state.route)

    def test_live_retriever_uses_pipeline_and_graph(self) -> None:
        fake_pipeline = _FakePipeline()
        fake_graph = _FakeGraphStore()
        retriever = LiveIndexedRetriever(pipeline=fake_pipeline, graph_store=fake_graph)
        chunks, deps = retriever.index_document(
            repo="core",
            path="svc.py",
            owner_team="platform",
            classification="internal",
            acl_roles=("dev",),
            content="import os",
        )
        self.assertEqual(chunks, 3)
        self.assertEqual(deps, 2)
        result = retriever.retrieve(query="owner", repo="core", module_path="svc.py", limit=3)
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.dependencies, ["dep.alpha", "dep.beta"])

    def test_chunker_falls_back_to_fixed_size_for_unsupported_file_type(self) -> None:
        chunker = TreeSitterChunker()
        content = ("a" * 50) + "\n" + ("b" * 50)
        chunks = chunker.chunk("README.md", content, chunk_size=40)
        self.assertEqual(chunks, [content[i : i + 40] for i in range(0, len(content), 40)])

    def test_local_mode_uses_in_memory_vector_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = str(Path(tmp_dir) / "graph.sqlite")
            settings = AppSettings(providers=ProviderSettings(local_mode=True, dependency_graph_db_path=db_path))
            retriever = LiveIndexedRetriever(settings=settings)
            self._assert_local_mode_round_trip(retriever)

    def _assert_local_mode_round_trip(self, retriever: LiveIndexedRetriever) -> None:
        retriever.index_document(
            repo="local",
            path="svc.py",
            owner_team="platform",
            classification="internal",
            acl_roles=("dev",),
            content="import os\n\ndef f():\n    return 1\n",
        )
        result = retriever.retrieve(query="who owns svc", repo="local", module_path="svc.py")
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.dependencies, ["os"])

    def test_langgraph_state_machine_routes_by_execution_path(self) -> None:
        orchestrator = PhaseOrchestrator()
        graph = build_langgraph_state_machine(orchestrator)
        result = graph.invoke(
            OrchestratorState(
                user_query="incident",
                task_type="incident_rca",
                risk_level="high",
                confidence=0.5,
                evidence_coverage=0.4,
            )
        )
        self.assertEqual(result["route"]["execution_path"], "deep")
        self.assertIn("Deep investigation", result["response"])

    def test_tree_sitter_chunker_uses_ast_segments(self) -> None:
        source = "import os\n\ndef hello():\n    return 1\n"
        root = _FakeNode(
            "module",
            0,
            len(source),
            [_FakeNode("import_statement", 0, 9), _FakeNode("function_definition", 11, len(source))],
        )
        chunker = _ChunkerForTest(_FakeParser(root))
        chunks = chunker.chunk("module.py", source, chunk_size=30)
        self.assertEqual(len(chunks), 2)
        self.assertIn("import os", chunks[0])


if __name__ == "__main__":
    unittest.main()
