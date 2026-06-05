"""Reporter interface contract + pipeline progress bridge (no rich / no models)."""

from voxweave import pipeline
from voxweave.progress import Reporter


class _RecordingReporter(Reporter):
    """Records task/advance calls to verify countable-stage bridging."""

    def __init__(self) -> None:
        self.tasks: list[tuple[str, int]] = []
        self.advances = 0

    def task(self, label: str, total: int) -> None:
        self.tasks.append((label, total))

    def advance(self, n: int = 1) -> None:
        self.advances += n


def test_base_reporter_methods_are_noops():
    # base class is all no-ops, no rich dependency; library callers can pass a bare Reporter
    rep = Reporter()
    rep.stage("x")
    rep.task("y", 5)
    rep.advance()
    rep.chunks(3)
    rep.chunk_done()  # must not raise


def test_chunks_chunk_done_delegate_to_task_advance():
    rep = _RecordingReporter()
    rep.chunks(4)
    rep.chunk_done()
    rep.chunk_done()
    assert rep.tasks == [("per-chunk ASR+align", 4)]
    assert rep.advances == 2


def test_progress_bridge_starts_task_once_then_advances():
    # backend/songdet sequential callbacks (done=1,2,3, total=3) -> first call creates task, subsequent calls advance(1)
    rep = _RecordingReporter()
    cb = pipeline._progress_bridge(rep, "人声分离 (Roformer)")
    cb(1, 3)
    cb(2, 3)
    cb(3, 3)
    assert rep.tasks == [("人声分离 (Roformer)", 3)]  # task created only once
    assert rep.advances == 3  # +1 per window, reaches 3/3
