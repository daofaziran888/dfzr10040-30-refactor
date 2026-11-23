"""批次游标与确认依据：回填与重放共用的断点步进、游标更新与批次确认核对。

回填任务与检查点重放各自维护批次断点，本模块把两边重复的批次步进、
游标解析/推进以及接收方批次确认依据的核对收拢到一处：

- 批次步进：按固定段长把 [起点, 总量) 切成连续行区间；
- 游标更新：断点位置的解析与续办起点计算，已确认的数据不重复纳入；
- 确认依据：接收方对行区间的写入确认，按来源、行边界和内容摘要核对，
  任何一项不符都不允许两边跳过对应行。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_position(raw: Any) -> int:
    """解析断点游标：非整数值一律回退到 0，负值按 0 处理。"""
    try:
        return max(0, int(raw))
    except (ValueError, TypeError):
        return 0


def digest_content(content: Any) -> str:
    """内容摘要：对批次内容生成稳定摘要，作为确认依据的内容核对项。"""
    payload = json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def plan_segments(start: int, total: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """批次步进：从 start 起按 batch_size 依次产出 [seg_start, seg_end) 行区间。"""
    current = max(0, start)
    while current < total:
        seg_end = min(current + batch_size, total)
        yield current, seg_end
        current = seg_end


def effective_start(saved_position: Any, confirmed_boundary: int) -> int:
    """续办起点：断点游标与实际确认边界取较大者。

    已确认的数据不重复纳入；未经确认的行不允许跳过。
    """
    return max(parse_position(saved_position), confirmed_boundary)


@dataclass
class BatchConfirmation:
    """批次确认依据：接收方对一段行范围写入结果的确认。"""
    source: str
    start: int
    end: int
    digest: str
    confirmed_at: str = field(default_factory=_now_iso)

    def to_record(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "digest": self.digest,
            "confirmed_at": self.confirmed_at,
        }


def confirmation_from_payload(payload: Any) -> BatchConfirmation | None:
    """把回调返回的确认依据字典解析为 BatchConfirmation；非法负载返回 None。"""
    if not isinstance(payload, dict):
        return None
    try:
        source = str(payload["source"])
        start = int(payload["start"])
        end = int(payload["end"])
        digest = str(payload["digest"])
    except (KeyError, TypeError, ValueError):
        return None
    return BatchConfirmation(source=source, start=start, end=end, digest=digest)


class ConfirmationLedger:
    """确认依据台账：按任务登记、核对与撤销批次确认，计算实际确认的行边界。"""

    def __init__(self) -> None:
        self._confirmations: dict[str, list[BatchConfirmation]] = {}
        self._sources: dict[str, str] = {}

    # ---- 来源核对 ----

    def bind_source(self, task_code: str, source: str) -> str:
        """登记任务的接收方来源；已绑定时返回既有来源，保持两个入口一致。"""
        bound = self._sources.get(task_code)
        if bound is None:
            self._sources[task_code] = source
            return source
        return bound

    def expected_source(self, task_code: str) -> str | None:
        return self._sources.get(task_code)

    # ---- 登记与核对 ----

    def record(self, task_code: str, confirmation: BatchConfirmation, *,
               expected_start: int, expected_end: int,
               expected_digest: str) -> tuple[bool, str]:
        """核对并登记确认依据。

        来源、行边界或内容摘要任何一项不符都拒绝登记，
        对应行不得被任何一侧跳过。
        """
        expected_source = self._sources.get(task_code)
        if expected_source is not None and confirmation.source != expected_source:
            return False, (
                f"确认来源不符：期望「{expected_source}」，"
                f"收到「{confirmation.source}」"
            )
        if confirmation.start != expected_start or confirmation.end != expected_end:
            return False, (
                f"确认行边界不符：期望 [{expected_start}, {expected_end})，"
                f"收到 [{confirmation.start}, {confirmation.end})"
            )
        if confirmation.digest != expected_digest:
            return False, "确认内容摘要不符"
        if expected_source is None:
            self._sources[task_code] = confirmation.source
        self._confirmations.setdefault(task_code, []).append(confirmation)
        return True, "确认依据已登记"

    # ---- 边界与待处理 ----

    def confirmed_boundary(self, task_code: str) -> int:
        """实际确认的行边界：有效确认自 0 起连续覆盖的终点。"""
        boundary = 0
        for start, end in sorted(
            (c.start, c.end) for c in self._confirmations.get(task_code, [])
        ):
            if start <= boundary < end:
                boundary = end
        return boundary

    def pending_ranges(self, task_code: str, total: int) -> list[tuple[int, int]]:
        """实际待处理的行区间：[0, total) 中未被有效确认覆盖的部分。"""
        pending: list[tuple[int, int]] = []
        cursor = 0
        for start, end in sorted(
            (c.start, c.end) for c in self._confirmations.get(task_code, [])
        ):
            start = max(0, start)
            end = min(end, total)
            if start > cursor:
                pending.append((cursor, start))
            cursor = max(cursor, end)
            if cursor >= total:
                break
        if cursor < total:
            pending.append((cursor, total))
        return pending

    def is_complete(self, task_code: str, total: int) -> bool:
        """结案判定：实际确认边界覆盖全部行。"""
        return self.confirmed_boundary(task_code) >= total

    # ---- 撤销 ----

    def revoke(self, task_code: str, start: int, end: int) -> bool:
        """接收方撤销一段确认：只重开对应行，相交的确认按边界裁开。"""
        confirmations = self._confirmations.get(task_code)
        if not confirmations:
            return False
        revised: list[BatchConfirmation] = []
        changed = False
        for conf in confirmations:
            if conf.end <= start or conf.start >= end:
                revised.append(conf)
                continue
            changed = True
            if conf.start < start:
                revised.append(BatchConfirmation(
                    source=conf.source, start=conf.start, end=start,
                    digest=conf.digest, confirmed_at=conf.confirmed_at))
            if conf.end > end:
                revised.append(BatchConfirmation(
                    source=conf.source, start=end, end=conf.end,
                    digest=conf.digest, confirmed_at=conf.confirmed_at))
        if changed:
            self._confirmations[task_code] = revised
        return changed

    # ---- 查询 ----

    def confirmations(self, task_code: str) -> list[dict[str, Any]]:
        """已登记确认依据清单，供两个入口核对与审计。"""
        return [c.to_record() for c in self._confirmations.get(task_code, [])]


# 全局单例：回填与重放两个入口共用的确认依据台账
confirmation_ledger = ConfirmationLedger()
