"""Pipeline run bookkeeping.

A Run row is the durable record of one pipeline execution. Its id is the
same value bound as run_id in core.logging, so a row here and every log
line from that execution share one identifier - which is how you pull a
single run out of interleaved logs in P8.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from lodestar.storage.models import Run, RunStatus


class RunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def start(self, trigger: str = "manual") -> Run:
        """Record that a run has begun. Status is RUNNING until told otherwise.

        Writing the row up front, rather than at the end, means a run that
        crashes hard still leaves evidence - it sits in RUNNING forever,
        which is a visible, diagnosable state. A row only written on success
        makes a crashed run indistinguishable from one that never started.
        """
        run = Run(id=uuid.uuid4(), trigger=trigger, status=RunStatus.RUNNING)
        self.session.add(run)
        self.session.flush()  # assign defaults without ending the transaction
        return run

    def finish(self, run_id: uuid.UUID, stats: dict[str, Any]) -> bool:
        run = self.session.get(Run, run_id)
        if run is None:
            return False
        run.status = RunStatus.SUCCEEDED
        run.finished_at = datetime.now(UTC)
        run.stats = stats
        return True

    def fail(
        self, run_id: uuid.UUID, error: str, stats: dict[str, Any] | None = None
    ) -> bool:
        run = self.session.get(Run, run_id)
        if run is None:
            return False
        run.status = RunStatus.FAILED
        run.finished_at = datetime.now(UTC)
        run.error = error[:8000]  # a traceback can be enormous
        if stats is not None:
            run.stats = stats
        return True

    def get(self, run_id: uuid.UUID) -> Run | None:
        return self.session.get(Run, run_id)

    def list_recent(self, limit: int = 20) -> list[Run]:
        statement = select(Run).order_by(Run.started_at.desc()).limit(limit)
        return list(self.session.execute(statement).scalars().all())

    def last_successful(self) -> Run | None:
        """The previous good run - gives ingestion its "fetch since" point."""
        statement = (
            select(Run)
            .where(Run.status == RunStatus.SUCCEEDED)
            .order_by(Run.started_at.desc())
            .limit(1)
        )
        return self.session.execute(statement).scalar_one_or_none()
