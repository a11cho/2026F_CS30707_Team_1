"""아카이브 정리 후에도 학습 계측과 보고서 경로 검증이 작동하는지 검사합니다."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from .paper_report.prepare_paper_data import verify_source_snapshot, find_previous_run


class MaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.repo=Path(self.temp.name)/"repo"
        self.module=self.repo/"smoke_test_mc_td"
        self.run=self.module/"runs"/"example"
        self.snapshot=self.run/"source_snapshot"/"smoke_test_mc_td"/"example.py"
        self.snapshot.parent.mkdir(parents=True)
        self.snapshot.write_bytes(b"original source\n")
        self.active=self.module/"example.py"
        self.active.write_bytes(self.snapshot.read_bytes())
        manifest={"source_sha256":{"smoke_test_mc_td\\example.py":hashlib.sha256(self.snapshot.read_bytes()).hexdigest()}}
        (self.run/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_snapshot_is_authoritative_and_active_changes_are_reported(self):
        result=verify_source_snapshot(self.run,self.repo)
        self.assertTrue(result["snapshot_verified"])
        self.assertEqual(result["current_source_changes"],[])
        self.active.write_bytes(b"new maintained source\n")
        result=verify_source_snapshot(self.run,self.repo)
        self.assertEqual(result["current_source_changes"][0]["status"],"changed")

    def test_missing_old_active_source_does_not_block_snapshot_verification(self):
        # 실제 파일 삭제 대신 존재하지 않는 현재 repo 경로로 검사합니다.
        alternate=self.repo/"another_current_repo"
        alternate.mkdir()
        result=verify_source_snapshot(self.run,alternate)
        self.assertTrue(result["snapshot_verified"])
        self.assertEqual(result["current_source_changes"][0]["status"],"moved_or_missing")

    def test_corrupted_snapshot_is_rejected(self):
        self.snapshot.write_bytes(b"corrupted snapshot\n")
        with self.assertRaises(ValueError):
            verify_source_snapshot(self.run,self.repo)

    def test_archive_lookup_precedes_legacy_path(self):
        run_id="20261031T093149737642Z"
        legacy=self.module/"runs"/run_id
        legacy.mkdir()
        self.assertEqual(find_previous_run(self.module),legacy)
        archived=self.module/"archive"/"legacy_20261031"/"runs"/run_id
        archived.mkdir(parents=True)
        self.assertEqual(find_previous_run(self.module),archived)

    def test_duplicate_archive_paths_are_rejected(self):
        for name in ("first","second"):
            (self.module/"archive"/name/"runs"/"20261031T093149737642Z").mkdir(parents=True)
        with self.assertRaises(ValueError):
            find_previous_run(self.module)

    @unittest.skipUnless(sys.platform=="win32","Windows 프로세스 메모리 검사")
    def test_current_memory_utility(self):
        from .runtime_metrics import memory
        values=memory()
        self.assertGreater(values["working_set_bytes"],0)
        self.assertGreaterEqual(values["peak_working_set_bytes"],values["working_set_bytes"])


if __name__=="__main__":
    unittest.main()
