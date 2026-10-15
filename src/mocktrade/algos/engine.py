"""The algo engine: parent orders worked by strategies through child orders.

    trader ─► start(code, instrument, side, qty, params) ─► parent (A1) ─► strategy
    strategy ─► ctx.send / ctx.modify / ctx.cancel ─► OrderManager (risk, throttle, broker)
    order events (parent_id = A1) ─► parent's filled/avg from its children ─► strategy.on_child
    prices ─► strategy.on_tick        heartbeat (every 100 ms) ─► strategy.on_timer
    pause ─► pull working children, stop sending     resume ─► strategy.on_resume
    stop / kill switch ─► pull every child ─► STOPPED once none is working

Rules (each one a real failure professional desks guard against):
- Children go through the same order manager, risk checks and throttle as manual orders;
  the throttle lets algo orders wait their turn instead of refusing them.
- A parent never over-sends: filled + working + a new child never exceeds its quantity.
- Its filled quantity is the sum of its children's own filled quantities (recomputed, never
  added up from events, so a repeated update can't double count).
- A child whose outcome is unknown holds the algo: nothing more is sent until it's answered.
- Strategy code that raises fails only its own algo; its children are pulled.
- A kill switch stops every algo of its account (the firm switch: every algo).
- Algos don't survive a restart: their leftover children are cancelled at start.
Every state change is audited in the event log and pushed to listeners (the screens).
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Protocol

from mocktrade.algos.spec import AlgoSpec, check_params
from mocktrade.domain.orders import OrderRequest, OrderType, Product, Side, Validity
from mocktrade.domain.ticks import Tick
from mocktrade.marketdata.service import MarketData
from mocktrade.orders.manager import OrderManager
from mocktrade.orders.model import EventKind, OrderError, OrderEvent, OrderView, Pending

log = logging.getLogger(__name__)

HEARTBEAT_S = 0.1
MAX_REJECTS_IN_A_ROW = 3
AUTO_PROTECTION = Decimal("-1")  # market children: the broker's automatic protection


class AlgoState(Enum):
    WORKING = "WORKING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"  # pulling its children
    DONE = "DONE"  # fully filled
    STOPPED = "STOPPED"  # stopped by the trader, a kill switch or the engine
    FAILED = "FAILED"  # its own code broke, or the broker kept refusing

    @property
    def is_final(self) -> bool:
        return self in (AlgoState.DONE, AlgoState.STOPPED, AlgoState.FAILED)


@dataclass(frozen=True)
class AlgoView:
    """One parent order as screens show it."""

    algo_id: str
    code: str
    account_id: str
    user_id: str
    instrument_id: str
    side: Side
    qty: int
    params: dict[str, Any]
    state: AlgoState
    filled_qty: int
    avg_price_paise: int
    working_qty: int
    children: tuple[str, ...]
    reason: str
    status: str  # what it's doing now, in words (from the strategy)
    extra: dict[str, Any]  # the strategy's own figures for the screen (e.g. a spread's legs)
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class AlgoEvent:
    kind: str  # STARTED, UPDATED, PAUSED, RESUMED, STOPPING, DONE, STOPPED, FAILED
    ts: datetime
    algo: AlgoView
    detail: str = ""


class Strategy(Protocol):
    """One running algo's logic. Every method may use `ctx` to look and act."""

    def start(self, ctx: "AlgoContext") -> None: ...
    def on_tick(self, ctx: "AlgoContext", tick: Tick) -> None: ...
    def on_child(self, ctx: "AlgoContext", child: OrderView, event: OrderEvent) -> None: ...
    def on_timer(self, ctx: "AlgoContext", now: datetime) -> None: ...
    def on_pause(self, ctx: "AlgoContext") -> None: ...
    def on_resume(self, ctx: "AlgoContext") -> None: ...


class BaseStrategy:
    """Do-nothing defaults, so a strategy only writes the methods it needs."""

    def start(self, ctx: "AlgoContext") -> None:
        pass

    def on_tick(self, ctx: "AlgoContext", tick: Tick) -> None:
        pass

    def on_child(self, ctx: "AlgoContext", child: OrderView, event: OrderEvent) -> None:
        pass

    def on_timer(self, ctx: "AlgoContext", now: datetime) -> None:
        pass

    def on_pause(self, ctx: "AlgoContext") -> None:
        pass

    def on_resume(self, ctx: "AlgoContext") -> None:
        pass

    def done(self, ctx: "AlgoContext") -> bool:
        """Finished when the main leg is fully filled (a spread also waits for its hedge)."""
        return ctx.filled(ctx.instrument_id, ctx.side) >= ctx.qty

    def keep_on_stop(self, ctx: "AlgoContext", child: OrderView) -> bool:
        """Children to leave working on pause/stop (a spread keeps its hedges)."""
        return False

    def on_finish(self, ctx: "AlgoContext") -> str | None:
        """A warning to add when the algo ends (e.g. a spread left unhedged), or None."""
        return None


StrategyFactory = Callable[[dict[str, Any]], Strategy]


@dataclass(frozen=True)
class AlgoType:
    """A kind of algo the engine can run: its spec (for the screens) and how to make one."""

    spec: AlgoSpec
    make: StrategyFactory
    exchange_algo_id: str | None = None  # the exchange's id once empanelled (SEBI framework)


class AlgoError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass
class _Parent:
    algo_id: str
    code: str
    account_id: str
    user_id: str
    instrument_id: str
    side: Side
    qty: int
    params: dict[str, Any]
    product: Product
    strategy: Strategy
    exchange_algo_id: str | None
    created_at: datetime
    updated_at: datetime
    state: AlgoState = AlgoState.WORKING
    reason: str = ""
    status: str = ""
    extra: dict[str, Any] = field(default_factory=dict)
    children: list[str] = field(default_factory=list)
    rejects_in_a_row: int = 0
    instruments: set[str] = field(default_factory=set)  # prices it wants (legs)
    final: AlgoState = AlgoState.STOPPED  # the state it ends in once its children are gone
    gave_up: bool = False  # failing: nothing more is sent, not even hedges
    # Sent but not yet seen by the order manager (its SUBMITTED event registers them):
    in_flight: int = 0  # how many child orders
    in_flight_qty: int = 0  # their quantity on the main leg (counts against the parent)


def _main(p: "_Parent", request: OrderRequest) -> bool:
    """On the main leg: the parent's instrument, on the parent's side. (A bracket's exits are
    on the same instrument but the other side; they don't count toward the parent.)"""
    return request.instrument_id == p.instrument_id and request.side is p.side


class AlgoContext:
    """What a strategy can see and do. Every order goes through the order manager."""

    def __init__(self, engine: "AlgoEngine", parent: _Parent) -> None:
        self._engine = engine
        self._parent = parent

    # --- looking ---
    @property
    def algo_id(self) -> str:
        return self._parent.algo_id

    @property
    def params(self) -> dict[str, Any]:
        return self._parent.params

    @property
    def instrument_id(self) -> str:
        return self._parent.instrument_id

    @property
    def side(self) -> Side:
        return self._parent.side

    @property
    def qty(self) -> int:
        return self._parent.qty

    def now(self) -> datetime:
        return self._engine.clock()

    def latest(self, instrument_id: str | None = None) -> Tick | None:
        return self._engine.market_data.latest(instrument_id or self._parent.instrument_id)

    def instrument(self, instrument_id: str | None = None):
        return self._engine.instruments(instrument_id or self._parent.instrument_id)

    def children(self) -> list[OrderView]:
        return self._engine.children(self._parent)

    def working(self) -> list[OrderView]:
        return [c for c in self.children() if not c.status.is_final]

    def filled(self, instrument_id: str | None = None, side: Side | None = None) -> int:
        """Quantity filled so far, on one leg and/or one side if given (always positive)."""
        return sum(
            c.filled_qty
            for c in self.children()
            if (instrument_id is None or c.request.instrument_id == instrument_id)
            and (side is None or c.request.side is side)
        )

    def average_price(
        self, instrument_id: str | None = None, side: Side | None = None
    ) -> int | None:
        """Average fill price on a leg and side (the main leg and the parent's side by
        default), or None before any fill."""
        leg = instrument_id or self._parent.instrument_id
        side = side or self._parent.side
        fills = [(c.filled_qty, c.avg_price_paise) for c in self.children()
                 if c.request.instrument_id == leg and c.request.side is side
                 and c.filled_qty]  # fmt: skip
        qty = sum(q for q, _ in fills)
        return round(sum(q * p for q, p in fills) / qty) if qty else None

    def remaining(self) -> int:
        """Main leg (the parent's instrument and side): quantity not yet filled nor working."""
        p = self._parent
        working = sum(c.leaves_qty for c in self.working() if _main(p, c.request))
        sent = p.in_flight_qty  # sent a moment ago, not registered yet
        return max(0, p.qty - self.filled(p.instrument_id, p.side) - working - sent)

    @property
    def sending(self) -> int:
        """Main-leg quantity sent a moment ago that the order manager hasn't registered yet."""
        return self._parent.in_flight_qty

    @property
    def held(self) -> bool:
        """A child's outcome is unknown: send nothing until the broker answers."""
        return any(c.pending is Pending.UNCONFIRMED for c in self.children())

    # --- acting ---
    def send(
        self,
        qty: int,
        price_paise: int | None,
        *,
        side: Side | None = None,
        instrument_id: str | None = None,
        order_type: OrderType = OrderType.LIMIT,
        validity: Validity = Validity.DAY,
        check_remaining: bool = True,
        hedge: bool = False,
    ) -> bool:
        """Send a child order. False (and nothing sent) if the algo isn't working, is held,
        or the child would take the main leg past the parent's quantity."""
        p = self._parent
        if qty <= 0 or p.state.is_final or p.gave_up:
            return False
        if not hedge and (p.state is not AlgoState.WORKING or self.held):
            return False  # hedges (risk-reducing) still go out while paused or stopping
        leg = instrument_id or p.instrument_id
        main = leg == p.instrument_id and (side or p.side) is p.side
        if check_remaining and main and qty > self.remaining():
            return False
        request = OrderRequest(
            client_order_id="",
            account_id=p.account_id,
            user_id=p.user_id,
            instrument_id=leg,
            side=side or p.side,
            qty=qty,
            order_type=order_type,
            product=p.product,
            validity=validity,
            price_paise=price_paise if order_type in (OrderType.LIMIT, OrderType.SL) else None,
            market_protection=(
                None if order_type in (OrderType.LIMIT, OrderType.SL) else AUTO_PROTECTION
            ),
            algo_id=p.exchange_algo_id,
            parent_id=p.algo_id,
        )
        p.in_flight += 1
        if main:
            p.in_flight_qty += qty
        self._engine.spawn(self._engine.manager.submit(request))
        return True

    def modify(
        self, client_order_id: str, *, price_paise: int | None = None, qty: int | None = None
    ):
        if self._parent.state is not AlgoState.WORKING:
            return
        manager = self._engine.manager
        self._engine.spawn(
            self._engine.quiet(
                lambda: manager.modify(client_order_id, price_paise=price_paise, qty=qty)
            )
        )

    def cancel(self, client_order_id: str) -> None:
        self._engine.spawn(self._engine.quiet(lambda: self._engine.manager.cancel(client_order_id)))

    def watch(self, *instrument_ids: str) -> None:
        """Get on_tick for these instruments too (e.g. a spread's other leg)."""
        self._parent.instruments.update(instrument_ids)
        self._engine.watch(instrument_ids)

    def report(self, **figures: Any) -> None:
        """Figures for the algo screen (e.g. legs filled, unhedged quantity)."""
        if any(self._parent.extra.get(k) != v for k, v in figures.items()):
            self._parent.extra.update(figures)
            self._engine.changed(self._parent, "UPDATED")

    def say(self, status: str) -> None:
        """What the algo is doing now, in words, for the algo screen."""
        if status != self._parent.status:
            self._parent.status = status
            self._engine.changed(self._parent, "UPDATED")

    def finish(self, reason: str = "") -> None:
        """The strategy has nothing more to do (its working children are pulled)."""
        self._engine.stop_parent(self._parent, reason or "finished", final=AlgoState.STOPPED)

    def fail(self, reason: str) -> None:
        self._engine.stop_parent(self._parent, reason, final=AlgoState.FAILED)


class AlgoEngine:
    def __init__(
        self,
        account_id: str,
        manager: OrderManager,
        market_data: MarketData,
        instruments: Callable[[str], Any],
        types: Iterable[AlgoType],
        *,
        blocked: Callable[[], str | None] = lambda: None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        audit: Callable[[AlgoEvent], None] | None = None,
        id_prefix: str = "A",
    ) -> None:
        self.account_id = account_id
        self.manager = manager
        self.market_data = market_data
        self.instruments = instruments
        self.types = {t.spec.code: t for t in types}
        self.clock = clock
        self._blocked = blocked
        self._audit = audit
        self._prefix = id_prefix
        self._next = 0
        self._parents: dict[str, _Parent] = {}
        self._listeners: list[Callable[[AlgoEvent], None]] = []
        self._tasks: set[asyncio.Task] = set()
        self._subscription = market_data.subscribe()
        manager.subscribe(self._on_order_event)

    # --- reading ----------------------------------------------------------------------------

    def specs(self) -> list[AlgoSpec]:
        return [t.spec for t in self.types.values()]

    def algos(self) -> list[AlgoView]:
        return [self._view(p) for p in self._parents.values()]

    def get(self, algo_id: str) -> AlgoView | None:
        p = self._parents.get(algo_id)
        return self._view(p) if p else None

    def subscribe(self, listener: Callable[[AlgoEvent], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    # --- the trader's controls --------------------------------------------------------------

    def start(
        self,
        code: str,
        *,
        user_id: str,
        instrument_id: str,
        side: Side,
        qty: int,
        params: dict[str, Any],
        product: Product = Product.NRML,
    ) -> AlgoView:
        """Start an algo. Raises AlgoError with every problem in words."""
        kind = self.types.get(code)
        if kind is None:
            raise AlgoError(f"there is no algo called {code!r}")
        if (why := self._blocked()) is not None:
            raise AlgoError(why)
        instrument = self.instruments(instrument_id)
        if instrument is None:
            raise AlgoError(f"unknown instrument {instrument_id}")
        problems: list[str] = []
        try:
            instrument.check_quantity(qty)
        except ValueError as exc:
            problems.append(str(exc))
        clean, more = check_params(kind.spec, params)
        problems += more
        if problems:
            raise AlgoError("; ".join(problems))
        try:
            strategy = kind.make(clean)
        except ValueError as exc:
            raise AlgoError(str(exc)) from None
        self._next += 1
        now = self.clock()
        parent = _Parent(
            algo_id=f"{self._prefix}{self._next}",
            code=code,
            account_id=self.account_id,
            user_id=user_id,
            instrument_id=instrument_id,
            side=side,
            qty=qty,
            params=clean,
            product=product,
            strategy=strategy,
            exchange_algo_id=kind.exchange_algo_id,
            created_at=now,
            updated_at=now,
            instruments={instrument_id},
        )
        self._parents[parent.algo_id] = parent
        self.watch([instrument_id])
        self.changed(parent, "STARTED", f"{side.value} {qty} {instrument_id} {clean}")
        self._call(parent, lambda ctx: parent.strategy.start(ctx))
        return self._view(parent)

    def pause(self, algo_id: str) -> AlgoView:
        p = self._live(algo_id)
        if p.state is AlgoState.WORKING:
            p.state = AlgoState.PAUSED
            self._pull(p)
            self.changed(p, "PAUSED", "working children pulled; nothing new is sent")
            self._call(p, lambda ctx: p.strategy.on_pause(ctx))
        return self._view(p)

    def resume(self, algo_id: str) -> AlgoView:
        p = self._live(algo_id)
        if p.state is AlgoState.PAUSED:
            if (why := self._blocked()) is not None:
                raise AlgoError(why)
            p.state = AlgoState.WORKING
            p.rejects_in_a_row = 0
            self.changed(p, "RESUMED")
            self._call(p, lambda ctx: p.strategy.on_resume(ctx))
        return self._view(p)

    def stop(self, algo_id: str, reason: str = "stopped by the trader") -> AlgoView:
        p = self._live(algo_id)
        self.stop_parent(p, reason, final=AlgoState.STOPPED)
        return self._view(p)

    def stop_all(self, reason: str) -> int:
        """Stop every algo that's still running (kill switches call this). Returns how many."""
        live = [
            p
            for p in self._parents.values()
            if not p.state.is_final and p.state is not AlgoState.STOPPING
        ]
        for p in live:
            self.stop_parent(p, reason, final=AlgoState.STOPPED)
        return len(live)

    async def cancel_orphans(self) -> list[str]:
        """At start: algos don't survive a restart, so their leftover children are cancelled."""
        orphans = [
            o.client_order_id
            for o in self.manager.working()
            if o.request.parent_id is not None and o.request.parent_id not in self._parents
        ]
        for client_order_id in orphans:
            await self.quiet(lambda cid=client_order_id: self.manager.cancel(cid))
        return orphans

    # --- running ------------------------------------------------------------------------------

    async def run(self) -> None:
        """Prices to strategies, and a heartbeat for their timers. Runs until cancelled."""
        loop = asyncio.get_running_loop()
        next_beat = loop.time()
        while True:
            timeout = max(0.0, next_beat - loop.time())
            try:
                ticks = await asyncio.wait_for(self._subscription.next_batch(), timeout)
            except TimeoutError:
                ticks = []
            for tick in ticks:
                self.on_tick(tick)
            if loop.time() >= next_beat:
                self.heartbeat()
                next_beat = loop.time() + HEARTBEAT_S

    def on_tick(self, tick: Tick) -> None:
        for p in list(self._parents.values()):
            if p.state is AlgoState.WORKING and tick.instrument_id in p.instruments:
                self._call(p, lambda ctx, t=tick, s=p.strategy: s.on_tick(ctx, t))

    def heartbeat(self) -> None:
        now = self.clock()
        for p in list(self._parents.values()):
            if p.state is AlgoState.WORKING:
                self._call(p, lambda ctx, s=p.strategy: s.on_timer(ctx, now))
            elif p.state is AlgoState.STOPPING:
                self._maybe_stopped(p)

    async def settle(self) -> None:
        """Wait for every order call the strategies made (tests)."""
        while pending := [t for t in self._tasks if not t.done()]:
            # only unfinished tasks: awaiting finished ones never yields to the event loop,
            # so their clean-up callbacks would never run and this would spin forever
            await asyncio.gather(*pending)

    # --- for the context and the strategies ----------------------------------------------------

    def spawn(self, coroutine) -> None:
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    @staticmethod
    async def quiet(make: Callable[[], Awaitable[Any]]) -> None:
        """Modify/cancel from an algo: a refusal (already filled, too late) is not an error.
        Takes a function that makes the call, so nothing is created if the task never runs."""
        with contextlib.suppress(OrderError):
            await make()

    def watch(self, instrument_ids: Iterable[str]) -> None:
        self._subscription.add(instrument_ids)

    def children(self, p: _Parent) -> list[OrderView]:
        return [v for cid in p.children if (v := self.manager.get(cid)) is not None]

    def changed(self, p: _Parent, kind: str, detail: str = "") -> None:
        p.updated_at = self.clock()
        event = AlgoEvent(kind, p.updated_at, self._view(p), detail)
        if self._audit is not None and kind != "UPDATED":
            self._audit(event)
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception:
                log.exception("algo listener failed")

    def stop_parent(self, p: _Parent, reason: str, *, final: AlgoState) -> None:
        if p.state.is_final or p.state is AlgoState.STOPPING:
            return
        p.reason = reason
        p.final = final
        p.gave_up = final is AlgoState.FAILED
        p.state = AlgoState.STOPPING
        self._pull(p)
        self.changed(p, "STOPPING", reason)
        self._maybe_stopped(p)

    # --- inside -------------------------------------------------------------------------------

    def _live(self, algo_id: str) -> _Parent:
        p = self._parents.get(algo_id)
        if p is None:
            raise AlgoError(f"no algo {algo_id}")
        if p.state.is_final:
            raise AlgoError(f"{algo_id} has already finished ({p.state.value.lower()})")
        return p

    def _pull(self, p: _Parent) -> None:
        ctx = AlgoContext(self, p)
        for child in self.children(p):
            if child.status.is_final or child.pending is Pending.CANCEL:
                continue
            try:
                keep = p.strategy.keep_on_stop(ctx, child)
            except Exception:
                log.exception("algo %s keep_on_stop failed", p.algo_id)
                keep = False
            if not keep:
                cid = child.client_order_id
                self.spawn(self.quiet(lambda cid=cid: self.manager.cancel(cid)))

    def _maybe_stopped(self, p: _Parent) -> None:
        """STOPPING ends once no child is working and none is still on its way in."""
        if p.state is not AlgoState.STOPPING or p.in_flight:
            return
        if not any(not c.status.is_final for c in self.children(p)):
            try:
                warning = p.strategy.on_finish(AlgoContext(self, p))
            except Exception:
                log.exception("algo %s on_finish failed", p.algo_id)
                warning = None
            if warning:
                p.reason = f"{p.reason}; {warning}" if p.reason else warning
                self.changed(p, "ALERT", warning)
            p.state = p.final
            self.changed(p, p.state.value, p.reason)

    def _call(self, p: _Parent, fn: Callable[[AlgoContext], None]) -> None:
        """Run strategy code; if it raises, fail that algo only."""
        try:
            fn(AlgoContext(self, p))
        except Exception as exc:
            log.exception("algo %s failed", p.algo_id)
            self.stop_parent(p, f"the algo's own code failed: {exc!r}", final=AlgoState.FAILED)

    def _on_order_event(self, event: OrderEvent) -> None:
        order = event.order
        if order is None or order.request.parent_id is None:
            return
        p = self._parents.get(order.request.parent_id)
        if p is None:
            return
        if event.kind is EventKind.SUBMITTED and order.client_order_id not in p.children:
            p.children.append(order.client_order_id)
            p.in_flight = max(0, p.in_flight - 1)
            if _main(p, order.request):
                p.in_flight_qty = max(0, p.in_flight_qty - order.request.qty)
            if p.state is not AlgoState.WORKING:
                # sent just before a pause or stop: pull it straight away
                self.spawn(self.quiet(lambda: self.manager.cancel(order.client_order_id)))
        if event.kind is EventKind.REJECTED:
            p.rejects_in_a_row += 1
            if p.rejects_in_a_row >= MAX_REJECTS_IN_A_ROW and not p.gave_up:
                why = f"{MAX_REJECTS_IN_A_ROW} child orders refused in a row; last: {order.reason}"
                if p.state is AlgoState.STOPPING:
                    # already stopping (e.g. hedges refused under a kill switch): give up
                    p.gave_up, p.final = True, AlgoState.FAILED
                    p.reason = f"{p.reason}; {why}" if p.reason else why
                else:
                    self.stop_parent(p, why, final=AlgoState.FAILED)
        elif event.kind in (EventKind.SENT, EventKind.FILL):
            p.rejects_in_a_row = 0
        if p.state.is_final:
            return
        if p.state in (AlgoState.PAUSED, AlgoState.STOPPING):
            # fills can still arrive (a cancel crossed a fill): let the strategy hedge them
            self._call(p, lambda ctx: p.strategy.on_child(ctx, order, event))
            self._maybe_stopped(p)
            self.changed(p, "UPDATED", event.kind.value)
            return
        if event.kind is EventKind.FILL and self._is_done(p):
            p.reason = "fully filled"
            p.final = AlgoState.DONE
            p.state = AlgoState.STOPPING
            self._pull(p)  # anything left working on the main leg is now surplus
            self.changed(p, "UPDATED", "fully filled")
            self._maybe_stopped(p)
            if p.state is AlgoState.STOPPING:
                return
        if p.state is AlgoState.WORKING:
            self._call(p, lambda ctx: p.strategy.on_child(ctx, order, event))
        self.changed(p, "UPDATED", event.kind.value)

    def _is_done(self, p: _Parent) -> bool:
        done = getattr(p.strategy, "done", None)
        if done is None:  # a strategy without BaseStrategy's defaults
            return self._main_filled(p) >= p.qty
        try:
            return bool(done(AlgoContext(self, p)))
        except Exception as exc:
            log.exception("algo %s done() failed", p.algo_id)
            self.stop_parent(p, f"the algo's own code failed: {exc!r}", final=AlgoState.FAILED)
            return False

    def _main_filled(self, p: _Parent) -> int:
        return sum(c.filled_qty for c in self.children(p) if _main(p, c.request))

    def _view(self, p: _Parent) -> AlgoView:
        main = [c for c in self.children(p) if _main(p, c.request)]
        filled = sum(c.filled_qty for c in main)
        value = sum(c.filled_qty * c.avg_price_paise for c in main)
        return AlgoView(
            algo_id=p.algo_id,
            code=p.code,
            account_id=p.account_id,
            user_id=p.user_id,
            instrument_id=p.instrument_id,
            side=p.side,
            qty=p.qty,
            params=dict(p.params),
            state=p.state,
            filled_qty=filled,
            avg_price_paise=round(value / filled) if filled else 0,
            working_qty=sum(c.leaves_qty for c in main if not c.status.is_final),
            children=tuple(p.children),
            reason=p.reason,
            status=p.status,
            extra=dict(p.extra),
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
