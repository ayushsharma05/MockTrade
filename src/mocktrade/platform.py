"""Putting it together: one AccountRuntime per broker account, inside one Platform.

    submit ─► OrderManager ─► gate = RiskGate then OrderThrottle ─► Broker
                 │ events
                 ├─► EventLog          (audit, restart)
                 └─► PositionBook ─► DailyLossWatch ─► kill switch
    market data ─► PositionBook.mark
    broker positions ◄─► PositionReconciler

The order of the gates matters: risk first (it refuses without using any of the
rate limit), then the throttle (which counts what it lets through).
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

from mocktrade.algos.engine import AlgoEngine, AlgoEvent, AlgoType
from mocktrade.broker.base import Broker
from mocktrade.broker.rules import order_spec
from mocktrade.domain.accounts import BrokerAccount
from mocktrade.domain.instruments import Instrument
from mocktrade.domain.order_spec import OrderSpec
from mocktrade.domain.ticks import Tick
from mocktrade.marketdata.service import MarketData
from mocktrade.orders.basket import Baskets
from mocktrade.orders.gate import Gates
from mocktrade.orders.manager import OrderManager
from mocktrade.orders.model import ClientIds
from mocktrade.positions.book import PositionBook
from mocktrade.positions.reconcile import PositionReconciler
from mocktrade.refdata.store import InstrumentStore
from mocktrade.risk.checks import RiskGate
from mocktrade.risk.kill_switch import KillSwitch
from mocktrade.risk.limits import RiskLimits
from mocktrade.risk.loss_watch import DailyLossWatch
from mocktrade.risk.throttle import OrderThrottle
from mocktrade.store.events import EventLog

ALGO_ENDS = ("DONE", "STOPPED", "FAILED", "LOST")  # audit kinds ALGO_<end>: no longer running


@dataclass
class AccountRuntime:
    account: BrokerAccount
    broker: Broker
    orders: OrderManager
    positions: PositionBook
    risk: RiskGate
    throttle: OrderThrottle
    switch: KillSwitch
    reconciler: PositionReconciler
    loss_watch: DailyLossWatch | None
    algos: AlgoEngine
    baskets: Baskets


@dataclass
class Platform:
    instruments: InstrumentStore
    market_data: MarketData
    firm_switch: KillSwitch
    event_log: EventLog | None = None
    accounts: dict[str, AccountRuntime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.event_log is not None:
            self.firm_switch.on_change(self.event_log.record_kill)

    def account(self, account_id: str) -> AccountRuntime | None:
        return self.accounts.get(account_id)

    def mark(self, tick: Tick) -> None:
        """Send a tick to every account's positions (market data listener)."""
        for runtime in self.accounts.values():
            runtime.positions.mark(tick)

    async def run(self, *, reconcile: bool = True) -> None:
        """Run every background loop until cancelled: each account's broker updates and
        position checks, and marking positions to the latest prices."""
        loops = [self._mark_loop()]
        for runtime in self.accounts.values():
            loops.append(runtime.orders.run())
            loops.append(self._run_algos(runtime))
            if reconcile:
                loops.append(runtime.reconciler.run())
        await asyncio.gather(*loops)

    def lost_protection(self, account_id: str) -> list[str]:
        """At start: protection algos (stops held by MockTrade) that were running when the
        server last stopped. They don't survive a restart, so each is named, in words, for
        an alert; and noted as lost in the audit, so it is reported once."""
        if self.event_log is None:
            return []
        log = self.event_log
        open_: dict[str, dict] = {}
        for e in log.notes(account_id, [f"ALGO_{k}" for k in ALGO_ENDS + ("STARTED",)]):
            algo_id = e.extra.get("algo_id", "")
            if e.kind == "ALGO_STARTED":
                if e.extra.get("family") == "protection":
                    open_[algo_id] = e.extra
            else:
                open_.pop(algo_id, None)
        lost = []
        for algo_id, a in open_.items():
            words = (
                f"{a.get('code')} {a.get('side')} {a.get('qty')} {a.get('instrument_id')} "
                f"({algo_id}, {a.get('user_id')})"
            )
            lost.append(words)
            log.record_note("ALGO_LOST", account_id, f"not running after a restart: {words}", a)
        return lost

    @staticmethod
    async def _run_algos(runtime: AccountRuntime) -> None:
        """Algos don't survive a restart: cancel their leftover children, then run."""
        await runtime.algos.cancel_orphans()
        await runtime.algos.run()

    async def _mark_loop(self) -> None:
        """Watch the prices of every instrument anyone holds; mark positions as they move."""
        subscription = self.market_data.subscribe()
        try:
            while True:
                held = {
                    p.instrument_id
                    for runtime in self.accounts.values()
                    for p in runtime.positions.positions()
                }
                subscription.add(held - subscription.instrument_ids)
                try:
                    batch = await asyncio.wait_for(subscription.next_batch(), 1.0)
                except TimeoutError:
                    continue
                for tick in batch:
                    self.mark(tick)
        finally:
            subscription.close()

    def _algo_engine(
        self,
        account_id: str,
        orders: OrderManager,
        switch: KillSwitch,
        types: tuple[AlgoType, ...],
        clock: Callable[[], datetime],
    ) -> AlgoEngine:
        """The account's algos: blocked while a kill switch is on, stopped when one engages,
        every state change audited."""

        def blocked() -> str | None:
            for s in (self.firm_switch, switch):
                if s.engaged:
                    return f"trading is stopped: {s.state.reason}"
            return None

        audit = None
        family = {t.spec.code: t.spec.family for t in types}
        if self.event_log is not None:
            log = self.event_log

            def audit(event: AlgoEvent) -> None:
                a = event.algo
                log.record_note(
                    f"ALGO_{event.kind}",
                    account_id,
                    event.detail,
                    {
                        "algo_id": a.algo_id,
                        "code": a.code,
                        "family": family.get(a.code, ""),
                        "instrument_id": a.instrument_id,
                        "side": a.side.value,
                        "qty": a.qty,
                        "filled_qty": a.filled_qty,
                        "params": a.params,
                        "user_id": a.user_id,
                    },
                    ts=event.ts,
                )

        engine = AlgoEngine(
            account_id,
            orders,
            self.market_data,
            self.instruments.get,
            types,
            blocked=blocked,
            clock=clock,
            audit=audit,
        )

        def on_switch(_name: str, state) -> None:
            if state.engaged:
                engine.stop_all(f"trading stopped: {state.reason}")

        switch.on_change(on_switch)
        self.firm_switch.on_change(on_switch)
        return engine

    def add_account(
        self,
        account: BrokerAccount,
        broker: Broker,
        *,
        limits: RiskLimits | None = None,
        ids: ClientIds,
        auto_kill_on_loss: bool = False,
        throttle: OrderThrottle | None = None,
        algo_types: tuple[AlgoType, ...] = (),
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> AccountRuntime:
        if account.id in self.accounts:
            raise ValueError(f"account {account.id} is already running")
        limits = limits or RiskLimits()
        switch = KillSwitch(f"account {account.id}", clock=clock)
        book = PositionBook(account.id, clock=clock)
        holder: dict[str, OrderManager] = {}  # the gate needs the manager it guards
        risk = RiskGate(
            limits,
            instruments=self.instruments.get,
            prices=self.market_data.latest,
            position=book.net_qty,
            working=lambda: holder["orders"].working(),
            day_pnl_paise=book.day_pnl_paise,
            switches=(self.firm_switch, switch),
            clock=clock,
            positions=book.net_by_instrument,
        )
        throttle = throttle or OrderThrottle()
        orders = OrderManager(
            account.id,
            broker,
            self.instruments.get,
            ids=ids,
            gate=Gates([risk, throttle]),
            spec=spec_of(broker),
            clock=clock,
        )
        holder["orders"] = orders
        orders.subscribe(book.on_order_event)
        if self.event_log is not None:
            orders.subscribe(self.event_log.record)
            switch.on_change(self.event_log.record_kill)
        loss_watch = None
        if auto_kill_on_loss:
            loss_watch = DailyLossWatch(
                book.day_pnl_paise, limits.max_daily_loss_paise, switch, [orders]
            )
            book.subscribe(loss_watch.check)
        reconciler = PositionReconciler(
            book.net_by_instrument, broker, account_id=account.id, clock=clock
        )
        if self.event_log is not None:
            log = self.event_log
            reconciler.subscribe(
                lambda what, m: log.record_note(
                    f"POSITION_MISMATCH_{what.upper()}",
                    account.id,
                    m.message,
                    {"instrument_id": m.instrument_id, "ours": m.ours, "broker": m.broker},
                )
            )
        algos = self._algo_engine(account.id, orders, switch, algo_types, clock)
        basket_audit = None
        if self.event_log is not None:
            log = self.event_log

            def basket_audit(kind: str, detail: str, extra: dict) -> None:
                log.record_note(kind, account.id, detail, extra)

        baskets = Baskets(
            account.id,
            orders,
            risk,
            broker,
            self.instruments.get,
            self.market_data.latest,
            audit=basket_audit,
        )
        runtime = AccountRuntime(
            account,
            broker,
            orders,
            book,
            risk,
            throttle,
            switch,
            reconciler,
            loss_watch,
            algos,
            baskets,
        )
        self.accounts[account.id] = runtime
        return runtime


def spec_of(broker: Broker) -> Callable[[Instrument], OrderSpec]:
    """The broker's own order spec if it publishes one, else the standard NSE/BSE one."""
    published = getattr(broker, "order_spec", None)
    return published if callable(published) else order_spec
