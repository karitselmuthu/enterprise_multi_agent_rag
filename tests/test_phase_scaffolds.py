import tempfile
import unittest
from pathlib import Path

from phase0.foundation.config import HardwareConfig
from phase0.phase1.ingestion import ChunkIndexer, IngestionPlanner
from phase0.phase3.security import AccessContext, EvidenceChunk, ImmutableAuditLog, RetrievalGuard
from phase0.phase5.rollout import ActionRequest, HumanApprovalGate


class PhaseScaffoldTests(unittest.TestCase):
    def test_ingestion_planner_prefers_remote_qdrant_on_8gb(self) -> None:
        planner = IngestionPlanner(HardwareConfig(system_ram_gb=8, available_model_ram_gb=4.0, prefer_remote_qdrant=True))
        self.assertTrue(planner.should_use_remote_vector_store())

    def test_secret_scan_rejects_sensitive_content(self) -> None:
        indexer = ChunkIndexer()
        with self.assertRaises(ValueError):
            indexer.build_chunks(
                repo="core",
                path="app.py",
                owner_team="platform",
                classification="internal",
                acl_roles=("dev",),
                content='API_KEY="abcd-secret-value"',
            )

    def test_retrieval_guard_applies_entitlements(self) -> None:
        guard = RetrievalGuard()
        context = AccessContext(
            user_id="u1",
            roles=("sre",),
            attributes={},
            repo_entitlements=("service-a",),
            clearance="internal",
        )
        chunks = [
            EvidenceChunk(
                chunk_id="1",
                repo="service-a",
                classification="internal",
                acl_roles=("sre",),
                text="allowed",
            ),
            EvidenceChunk(
                chunk_id="2",
                repo="service-b",
                classification="internal",
                acl_roles=("sre",),
                text="blocked",
            ),
        ]
        filtered = guard.filter_chunks(context, chunks)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].chunk_id, "1")

    def test_audit_log_is_hash_chained(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "audit.jsonl"
            audit_log = ImmutableAuditLog(log_path)
            audit_log.append({"query": "hello"})
            audit_log.append({"query": "world"})
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn('"prev_hash": "GENESIS"', lines[0])

    def test_human_approval_gate(self) -> None:
        gate = HumanApprovalGate()
        self.assertTrue(
            gate.requires_approval(
                ActionRequest(category="production_change", description="deploy", requested_by="u1")
            )
        )
        self.assertFalse(
            gate.requires_approval(
                ActionRequest(category="ownership_lookup", description="query", requested_by="u1")
            )
        )


if __name__ == "__main__":
    unittest.main()
