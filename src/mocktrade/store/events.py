"""The event log: every order event, append-only, in a crash-safe SQLite file.

    order manager ─► record(event) ─► in-memory queue ─► writer thread ─► SQLite (WAL)
       (never waits for disk)                            (batches, retries)

    screens / restart ─► history(), events(), latest_views(), last_counter()

Why these choices:
- A separate writer thread: the trading loop never waits for the disk.
- Batches: one transaction for many events, so bursts stay cheap.
- WAL + synchronous=NORMAL: nothing is lost if the app crashes; a power cut can
  lose the last few milliseconds. The broker's own order book is the final truth,
  and reconciliation repairs the rest.
- If writing fails (disk full, file locked), events stay in memory and are retried,
  never dropped. `healthy` turns False and `on_failure` is called once per outage,
  on the event loop, so the app can alert or stop trading.
- Python's built-in sqlite3: no extra dependency. Moving to PostgreSQL later means
  another class with the same methods.
"""

import asyncio
import json
import logging
import queue
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mocktrade.orders.codec import (
    SCHEMA,
    fill_from_dict,
    fill_to_dict,
    view_from_dict,
    view_to_dict,
)
from mocktrade.orders.model import EventKind, Fill, OrderEvent, OrderView
from mocktrade.risk.kill_switch import KillState

log = logging.getLogger(__name__)

KILL_SWITCH = "KILL_SWITCH"  # event kind for kill switch changes
_NO_ORDER = (EventKind.EXTERNAL.value, KILL_SWITCH)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    account_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    client_order_id TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_by_order ON events (client_order_id, seq);
CREATE INDEX IF NOT EXISTS events_by_account ON events (account_id, client_order_id, seq);
"""

_INSERT = (
    "INSERT INTO events (ts, account_id, kind, client_order_id, detail, payload) "
    "VALUES (?, ?, ?, ?, ?, ?)"
)


@dataclass(frozen=True)
class StoredEvent:
    seq: int  # position in the log: later events have bigger numbers
    ts: datetime
    account_id: str
    kind: str  # an EventKind value, or KILL_SWITCH
    client_order_id: str
    detail: str
    order: OrderView | None
    fill: Fill | None
    extra: dict[str, Any]  # anything else (e.g. who engaged a kill switch)


@dataclass
class _Flush:
    future: asyncio.Future
    loop: asyncio.AbstractEventLoop


_STOP = object()


class EventLog:
    def __init__(
        self,
        path: str | Path,
        *,
        on_failure: Callable[[Exception], None] | None = None,
        batch_size: int = 500,
        max_backoff_s: float = 2.0,
    ) -> None:
        self.path = Path(path)
        self._on_failure = on_failure
        self._batch_size = batch_size
        self._max_backoff_s = max_backoff_s
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closed = False
        self.healthy = True
        self.written = 0
        with self._reader() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SCHEMA_SQL)

    # --- lifecycle --------------------------------------------------------------------------

    def start(self) -> None:
        """Start the writer thread. Call from the event loop (on_failure is sent there)."""
        if self._thread is not None:
            return
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        self._thread = threading.Thread(target=self._writer, name="event-log", daemon=True)
        self._thread.start()

    def close(self) -> None:
        """Write everything still queued, then stop. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        if self._thread is not None:
            self._queue.put(_STOP)
            self._thread.join()

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    # --- writing ----------------------------------------------------------------------------

    def record(self, event: OrderEvent) -> None:
        """Queue one order event. Never blocks; use as an order manager listener."""
        payload: dict[str, Any] = {"v": SCHEMA}
        if event.order is not None:
            payload["order"] = view_to_dict(event.order)
        if event.fill is not None:
            payload["fill"] = fill_to_dict(event.fill)
        self._put(
            (
                event.ts.isoformat(),
                event.account_id,
                event.kind.value,
                event.client_order_id,
                event.detail,
                json.dumps(payload, separators=(",", ":")),
            )
        )

    def record_kill(self, name: str, state: KillState) -> None:
        """Queue a kill switch change. Use as a KillSwitch listener."""
        at = state.at or datetime.now(UTC)
        payload = {"v": SCHEMA, "engaged": state.engaged, "by": state.by}
        detail = state.reason if state.engaged else "reset"
        self._put((at.isoformat(), name, KILL_SWITCH, "", detail, json.dumps(payload)))

    def record_note(
        self,
        kind: str,
        account_id: str,
        detail: str,
        extra: dict[str, Any] | None = None,
        ts: datetime | None = None,
    ) -> None:
        """Queue an audit note that isn't an order event, e.g. a position mismatch."""
        if kind in {k.value for k in EventKind}:
            raise ValueError(f"{kind} is an order event kind; use record()")
        payload = {"v": SCHEMA, **(extra or {})}
        stamp = (ts or datetime.now(UTC)).isoformat()
        self._put((stamp, account_id, kind, "", detail, json.dumps(payload)))

    async def flush(self) -> None:
        """Wait until everything queued so far is safely written."""
        if self._thread is None:
            raise RuntimeError("event log not started")
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._queue.put(_Flush(future, loop))
        await future

    def _put(self, row: tuple) -> None:
        if self._closed:
            raise RuntimeError("event log is closed")
        self._queue.put(row)

    def _writer(self) -> None:
        db = self._connect()
        db.execute("PRAGMA synchronous=NORMAL")
        rows: list[tuple] = []  # waiting to be written (kept if a write fails)
        flushes: list[_Flush] = []
        stopping = False
        backoff = 0.05
        try:
            while True:
                if not rows and not flushes and not stopping:
                    self._take(self._queue.get(), rows, flushes)  # sleep until there's work
                while len(rows) < self._batch_size:
                    try:
                        item = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    if item is _STOP:
                        stopping = True
                    else:
                        self._take(item, rows, flushes)
                if rows:
                    try:
                        self._write(db, rows)
                    except Exception as exc:
                        self._failed(exc)
                        if stopping and backoff >= self._max_backoff_s:
                            log.critical("event log closing with %d unwritten events", len(rows))
                            return
                        time.sleep(backoff)
                        backoff = min(backoff * 2, self._max_backoff_s)
                        continue
                    backoff = 0.05
                    self.written += len(rows)
                    rows.clear()
                    if not self.healthy:
                        log.warning("event log writing again")
                        self.healthy = True
                for flush in flushes:
                    flush.loop.call_soon_threadsafe(_resolve, flush.future)
                flushes.clear()
                if stopping and self._queue.empty():
                    return
        finally:
            db.close()

    def _take(self, item, rows: list, flushes: list) -> None:
        if item is _STOP:
            self._queue.put(_STOP)  # handled by the draining loop
        elif isinstance(item, _Flush):
            flushes.append(item)
        else:
            rows.append(item)

    def _write(self, db: sqlite3.Connection, rows: list[tuple]) -> None:
        with db:  # one transaction: all or nothing
            db.executemany(_INSERT, rows)

    def _failed(self, exc: Exception) -> None:
        if self.healthy:
            log.error("event log write failed, will retry: %r", exc)
            self.healthy = False
            if self._on_failure is not None:
                if self._loop is not None and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._on_failure, exc)
                else:
                    self._on_failure(exc)

    # --- reading ----------------------------------------------------------------------------

    def history(self, client_order_id: str) -> list[StoredEvent]:
        """Every event of one order, oldest first (the audit screen)."""
        return self._query(
            "SELECT * FROM events WHERE client_order_id = ? ORDER BY seq", (client_order_id,)
        )

    def events(
        self, *, after_seq: int = 0, account_id: str | None = None, limit: int = 1000
    ) -> list[StoredEvent]:
        """Events after `after_seq`, oldest first (screens catching up after a reconnect)."""
        if account_id is None:
            sql, args = "SELECT * FROM events WHERE seq > ? ORDER BY seq LIMIT ?", (after_seq,)
        else:
            sql = "SELECT * FROM events WHERE seq > ? AND account_id = ? ORDER BY seq LIMIT ?"
            args = (after_seq, account_id)
        return self._query(sql, (*args, limit))

    def notes(self, account_id: str, kinds: list[str]) -> list[StoredEvent]:
        """One account's notes of these kinds, oldest first (e.g. algo starts and ends)."""
        if not kinds:
            return []
        marks = ",".join("?" * len(kinds))
        sql = f"SELECT * FROM events WHERE account_id = ? AND kind IN ({marks}) ORDER BY seq"
        return self._query(sql, (account_id, *kinds))

    def page(
        self,
        names: list[str],
        *,
        before_seq: int | None = None,
        after_seq: int | None = None,
        limit: int = 200,
    ) -> list[StoredEvent]:
        """One page of the audit trail for these log names (an account id, its kill
        switch "account <id>", "firm"), **newest first**.

        - neither: the newest `limit` events (opening the screen)
        - `before_seq`: the `limit` events just older than it (scrolling back)
        - `after_seq`: every event newer than it, up to `limit` (live updates)
        """
        if not names:
            return []
        marks = ",".join("?" * len(names))
        if after_seq is not None:
            sql = f"""SELECT * FROM (
                SELECT * FROM events WHERE seq > ? AND account_id IN ({marks})
                ORDER BY seq LIMIT ?) ORDER BY seq DESC"""
            return self._query(sql, (after_seq, *names, limit))
        top = before_seq if before_seq is not None else 2**62
        sql = f"""SELECT * FROM events WHERE seq < ? AND account_id IN ({marks})
                  ORDER BY seq DESC LIMIT ?"""
        return self._query(sql, (top, *names, limit))

    def latest_views(self, account_id: str, *, since: datetime | None = None) -> list[OrderView]:
        """Each order's newest known state, oldest order first (rebuilding after a restart).

        `since` keeps only orders created at or after it (e.g. today's session start)."""
        sql = f"""
            SELECT * FROM events WHERE seq IN (
                SELECT MAX(seq) FROM events
                WHERE account_id = ? AND client_order_id != ''
                  AND kind NOT IN ({",".join("?" * len(_NO_ORDER))})
                GROUP BY client_order_id
            ) ORDER BY seq
        """
        stored = self._query(sql, (account_id, *_NO_ORDER))
        views = [e.order for e in stored if e.order is not None]
        if since is not None:
            views = [v for v in views if v.created_at >= since]
        return sorted(views, key=lambda v: (v.created_at, v.client_order_id))

    def last_counter(self, prefix: str) -> int:
        """The highest number used after `prefix` in our client ids (0 if none), so a
        restarted app continues the sequence instead of reusing ids."""
        if not prefix.isalnum():
            raise ValueError("prefix must be letters or digits")
        with self._reader() as db:
            rows = db.execute(
                "SELECT DISTINCT client_order_id FROM events WHERE client_order_id LIKE ?",
                (prefix + "%",),
            ).fetchall()
        numbers = [int(cid[len(prefix) :]) for (cid,) in rows if cid[len(prefix) :].isdigit()]
        return max(numbers, default=0)

    def _query(self, sql: str, args: tuple) -> list[StoredEvent]:
        with self._reader() as db:
            db.row_factory = sqlite3.Row
            return [_stored(row) for row in db.execute(sql, args)]

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        """A short-lived connection that is always closed (sqlite3's own `with` only
        commits; it doesn't close). WAL lets readers run while the writer writes."""
        db = self._connect()
        try:
            with db:
                yield db
        finally:
            db.close()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)


def _stored(row: sqlite3.Row) -> StoredEvent:
    payload = json.loads(row["payload"])
    order = payload.pop("order", None)
    fill = payload.pop("fill", None)
    payload.pop("v", None)
    return StoredEvent(
        seq=row["seq"],
        ts=datetime.fromisoformat(row["ts"]),
        account_id=row["account_id"],
        kind=row["kind"],
        client_order_id=row["client_order_id"],
        detail=row["detail"],
        order=view_from_dict(order) if order is not None else None,
        fill=fill_from_dict(fill) if fill is not None else None,
        extra=payload,
    )


def _resolve(future: asyncio.Future) -> None:
    if not future.done():
        future.set_result(None)
