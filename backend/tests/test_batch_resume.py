"""回填与重放共用批次断点：确认依据核对与续办范围测试。

覆盖：
- 两个入口从断点 100 继续时首先处理的行；
- 首段已写而断点未更新时的续办范围（已确认数据不重复纳入）；
- 回填/重放交叉续办、不符确认（来源/行边界/内容）的拒绝；
- 完工后撤销末段确认的重开与更正行重写。
"""
from __future__ import annotations

import pytest

from app.batch_cursor import (
    BatchConfirmation,
    ConfirmationLedger,
    digest_content,
)
from app.fusion_engine import FusionEngine
from app.recovery import CheckpointManager

TOTAL = 205
BATCH = 100
TABLE = "WT_RESUME"


@pytest.fixture
def ledger() -> ConfirmationLedger:
    """每个测试获得独立台账，回填与重放通过注入共用。"""
    return ConfirmationLedger()


@pytest.fixture
def engine(ledger: ConfirmationLedger) -> FusionEngine:
    return FusionEngine(ledger=ledger)


@pytest.fixture
def checkpoints(ledger: ConfirmationLedger) -> CheckpointManager:
    return CheckpointManager(ledger=ledger)


def _records(total: int = TOTAL) -> list[dict]:
    return [{"id": f"row-{i}", "value": i} for i in range(total)]


def _make_task(engine: FusionEngine):
    """正常每段大小 100：parallel_partitions=1 -> batch_size=100。"""
    return engine.create_backfill_task(
        target_wide_table=TABLE,
        backfill_range="2026-01-01~2026-06-30",
        parallel_partitions=1,
    )


def _segments(ledger: ConfirmationLedger, task_code: str) -> list[tuple[int, int]]:
    return [(c["start"], c["end"]) for c in ledger.confirmations(task_code)]


class TestResumeFromCheckpoint:
    """值班从原断点重试：两个入口从断点 100 继续。"""

    def test_backfill_resumes_from_checkpoint_100(
        self, engine: FusionEngine, ledger: ConfirmationLedger,
    ) -> None:
        task = _make_task(engine)
        task.checkpoint_position = "100"  # 原断点

        result = engine.execute_backfill_with_checkpoint(task.task_id, _records())

        assert result["ok"] is True
        assert result["batch_processed"] == 100
        assert result["checkpoint"] == "200"
        # 首先处理的是第 100 行（0 起），本批覆盖 [100, 200)
        assert _segments(ledger, task.task_code) == [(100, 200)]

    def test_replay_resumes_from_checkpoint_100(
        self, checkpoints: CheckpointManager,
    ) -> None:
        checkpoints.save_checkpoint("TASK-205", "100")
        calls: list[tuple[int, int]] = []

        def replay_fn(batch: list[int], start: int) -> dict:
            calls.append((start, batch[0]))
            return {"ok": True}

        result = checkpoints.replay_from_checkpoint(
            "TASK-205", replay_fn, total_items=TOTAL, batch_size=BATCH)

        assert result["ok"] is True
        assert result["complete"] is True
        # 首先回调的是第 100 行，随后按段推进
        assert calls[0] == (100, 100)
        assert [start for start, _ in calls] == [100, 200]


class TestWrittenButCursorStale:
    """首段已写而断点未更新：续办不重复纳入已确认的数据。"""

    def test_backfill_resume_skips_confirmed(
        self, engine: FusionEngine, ledger: ConfirmationLedger,
    ) -> None:
        task = _make_task(engine)
        records = _records()
        first = engine.execute_backfill_with_checkpoint(task.task_id, records)
        assert first["checkpoint"] == "100"
        # 模拟接续作业在游标落盘前停下：确认仍在，断点丢失
        task.checkpoint_position = ""

        resumed = engine.execute_backfill_with_checkpoint(task.task_id, records)

        assert resumed["batch_processed"] == 100
        assert resumed["checkpoint"] == "200"
        assert resumed["remaining"] == TOTAL - 200
        # 已确认的首段没有重复写入
        assert _segments(ledger, task.task_code) == [(0, 100), (100, 200)]

    def test_replay_resume_skips_confirmed(
        self,
        engine: FusionEngine,
        checkpoints: CheckpointManager,
        ledger: ConfirmationLedger,
    ) -> None:
        task = _make_task(engine)
        engine.execute_backfill_with_checkpoint(task.task_id, _records())
        task.checkpoint_position = ""  # 断点未更新，仅确认依据可查

        calls: list[int] = []

        def replay_fn(batch: list[int], start: int) -> dict:
            calls.append(start)
            return {"ok": True}

        result = checkpoints.replay_from_checkpoint(
            task.task_code, replay_fn, total_items=TOTAL, batch_size=BATCH)

        assert result["complete"] is True
        # 重放读取回填接收方的确认依据，从实际确认边界 100 起回调
        assert calls == [100, 200]
        assert ledger.confirmed_boundary(task.task_code) == TOTAL


class TestConfirmationVerification:
    """来源、行边界或内容不符的确认不能让两边跳过任何一行。"""

    @pytest.mark.parametrize("case", ["source", "boundary", "digest"])
    def test_invalid_confirmation_rejected(
        self,
        checkpoints: CheckpointManager,
        ledger: ConfirmationLedger,
        case: str,
    ) -> None:
        task_code = f"TASK-VERIFY-{case}"
        ledger.bind_source(task_code, TABLE)  # 接收方来源已建立
        checkpoints.save_checkpoint(task_code, "0")

        def replay_fn(batch: list[int], start: int) -> dict:
            confirmation = {
                "source": TABLE,
                "start": start,
                "end": start + len(batch),
                "digest": digest_content(batch),
            }
            if case == "source":
                confirmation["source"] = "WT_OTHER"
            elif case == "boundary":
                confirmation["start"] = start + 1
            else:
                confirmation["digest"] = "0" * 16
            return {"ok": True, "confirmation": confirmation}

        result = checkpoints.replay_from_checkpoint(
            task_code, replay_fn, total_items=TOTAL, batch_size=BATCH)

        assert result["ok"] is False
        assert result["complete"] is False
        # 没有一行被确认或跳过，游标停在原断点
        assert result["processed"] == 0
        assert ledger.confirmed_boundary(task_code) == 0
        assert ledger.pending_ranges(task_code, TOTAL) == [(0, TOTAL)]
        assert checkpoints.load_checkpoint(task_code)["position"] == "0"

    def test_ledger_record_checks_all_dimensions(self, ledger: ConfirmationLedger) -> None:
        ledger.bind_source("TASK-LEDGER", TABLE)
        good = BatchConfirmation(source=TABLE, start=0, end=100, digest="d" * 16)
        ok, _ = ledger.record(
            "TASK-LEDGER", good,
            expected_start=0, expected_end=100, expected_digest="d" * 16)
        assert ok is True

        bad_source = BatchConfirmation(source="WT_OTHER", start=100, end=200, digest="d" * 16)
        ok, message = ledger.record(
            "TASK-LEDGER", bad_source,
            expected_start=100, expected_end=200, expected_digest="d" * 16)
        assert ok is False and "来源" in message

        bad_boundary = BatchConfirmation(source=TABLE, start=100, end=201, digest="d" * 16)
        ok, message = ledger.record(
            "TASK-LEDGER", bad_boundary,
            expected_start=100, expected_end=200, expected_digest="d" * 16)
        assert ok is False and "行边界" in message

        bad_digest = BatchConfirmation(source=TABLE, start=100, end=200, digest="x" * 16)
        ok, message = ledger.record(
            "TASK-LEDGER", bad_digest,
            expected_start=100, expected_end=200, expected_digest="d" * 16)
        assert ok is False and "内容" in message

        # 只有有效确认计入边界
        assert ledger.confirmed_boundary("TASK-LEDGER") == 100


class TestCrossEntryResume:
    """回填与重放交叉续办：不符确认被拒绝，已提交范围互相识别。"""

    def test_mismatched_confirmation_rejected_then_resumed(
        self,
        engine: FusionEngine,
        checkpoints: CheckpointManager,
        ledger: ConfirmationLedger,
    ) -> None:
        task = _make_task(engine)
        records = _records()

        # 第一段 [0, 100) 由回填写入
        first = engine.execute_backfill_with_checkpoint(task.task_id, records)
        assert first["checkpoint"] == "100"

        replay_writes: list[int] = []

        def replay_fn(batch: list[int], start: int) -> dict:
            if start == 100:
                # 第二段 [100, 200)：接收方正常确认
                replay_writes.extend(batch)
                return {
                    "ok": True,
                    "confirmation": {
                        "source": TABLE,
                        "start": start,
                        "end": start + len(batch),
                        "digest": digest_content(batch),
                    },
                }
            # 末五行：返回一个来自不同范围的确认并中断
            return {
                "ok": True,
                "confirmation": {
                    "source": TABLE,
                    "start": 300,
                    "end": 305,
                    "digest": digest_content([300, 301, 302, 303, 304]),
                },
            }

        replay_result = checkpoints.replay_from_checkpoint(
            task.task_code, replay_fn, total_items=TOTAL, batch_size=BATCH)

        assert replay_result["ok"] is False
        assert replay_result["complete"] is False
        assert replay_result["processed"] == 200
        assert replay_writes == list(range(100, 200))
        # 不符的确认没有让游标越过末五行
        assert checkpoints.load_checkpoint(task.task_code)["position"] == "200"
        assert ledger.confirmed_boundary(task.task_code) == 200
        assert ledger.pending_ranges(task.task_code, TOTAL) == [(200, TOTAL)]

        # 回填再次续办：回填旧进度停在 100，但接续重放已提交的范围，
        # 应处理原来的末五行
        assert task.checkpoint_position == "100"
        resumed = engine.execute_backfill_with_checkpoint(task.task_id, records)
        assert resumed["batch_processed"] == 5
        assert resumed["remaining"] == 0
        assert resumed["complete"] is True
        assert _segments(ledger, task.task_code) == [(0, 100), (100, 200), (200, 205)]

        # 重放识别回填这次提交：不再触发任何回调
        def unexpected_call(batch: list[int], start: int) -> dict:
            raise AssertionError("已确认范围不应重复回调")

        final = checkpoints.replay_from_checkpoint(
            task.task_code, unexpected_call, total_items=TOTAL, batch_size=BATCH)
        assert final["ok"] is True
        assert final["complete"] is True
        assert final["processed"] == TOTAL

        # 最终的 205 行回填明细都各写一次
        covered: list[int] = []
        for conf in ledger.confirmations(task.task_code):
            covered.extend(range(conf["start"], conf["end"]))
        assert sorted(covered) == list(range(TOTAL))
        assert len(covered) == TOTAL
        assert replay_writes == list(range(100, 200))


class TestRevokeAndCorrect:
    """完工后撤销末段确认：恢复可续办，只重开末五行。"""

    def _complete_all(self, engine: FusionEngine, task, records: list[dict]) -> None:
        engine.execute_backfill_with_checkpoint(task.task_id, records)  # [0, 100)
        engine.execute_backfill_with_checkpoint(task.task_id, records)  # [100, 200)
        done = engine.execute_backfill_with_checkpoint(task.task_id, records)  # [200, 205)
        assert done["complete"] is True
        assert done["checkpoint"] == ""

    def test_revoke_reopens_only_last_rows(
        self,
        engine: FusionEngine,
        checkpoints: CheckpointManager,
        ledger: ConfirmationLedger,
    ) -> None:
        task = _make_task(engine)
        records = _records()
        self._complete_all(engine, task, records)
        assert ledger.is_complete(task.task_code, TOTAL) is True

        # 接收方撤销末段确认
        assert ledger.revoke(task.task_code, 200, TOTAL) is True

        # 结案恢复可续办状态：只重开末五行
        assert ledger.is_complete(task.task_code, TOTAL) is False
        assert ledger.pending_ranges(task.task_code, TOTAL) == [(200, TOTAL)]
        assert ledger.confirmed_boundary(task.task_code) == 200

        # 更正第 205 行后重新提交
        corrected = _records()
        corrected[204] = {"id": "row-204", "value": "更正后的第205行"}
        result = engine.execute_backfill_with_checkpoint(task.task_id, corrected)

        assert result["batch_processed"] == 5
        assert result["complete"] is True
        assert result["remaining"] == 0

        # 新提交把更正后的第 205 行写入
        tail = [c for c in ledger.confirmations(task.task_code) if c["start"] == 200]
        assert len(tail) == 1
        assert tail[0]["digest"] == digest_content(corrected[200:205])
        assert tail[0]["digest"] != digest_content(records[200:205])

        # 前 200 行不再写：确认段仍只有最初的两段，结案恢复
        early = [c for c in ledger.confirmations(task.task_code) if c["start"] < 200]
        assert [(c["start"], c["end"]) for c in early] == [(0, 100), (100, 200)]
        assert ledger.is_complete(task.task_code, TOTAL) is True

    def test_revoke_without_confirmation_is_noop(
        self, ledger: ConfirmationLedger,
    ) -> None:
        assert ledger.revoke("TASK-NONE", 0, 100) is False
