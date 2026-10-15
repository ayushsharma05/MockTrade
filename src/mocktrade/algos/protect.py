"""Protection held by MockTrade: trailing stop, OCO exit and bracket.

The stop of each is **synthetic**: MockTrade watches the last traded price and sends the exit
when the stop is touched. Nothing sits at the exchange for it until then, so a stop and a
target can never both fill and turn an exit into a new position the other way.

    trailing stop  the stop follows the best price by points or a percentage, never back;
                   touched ─► exit everything (market, or a limit some ticks past the stop)
    OCO exit       a target (a resting limit order) and a stop for a position;
                   target fills ─► done          stop touched ─► cancel the target, then exit
                   what is still open (the engine never lets the exits add up to more)
    bracket        an entry (limit, or market); each entry fill gets a target for its
                   quantity at once; the stop covers whatever is open; stop touched ─►
                   cancel the rest of the entry and the targets, then exit what is open

Honest about what it is: held by MockTrade, so it works only while the server runs (a
broker GTT, K9.3, survives our outages). On a restart these algos are gone: the platform
raises an alert naming each one, so a trader never believes a position is protected when
it isn't. Pause and stop pull every child, so the screen says the position is unprotected.

Prices: a sell stop is touched when the last price is at or below it; a buy stop at or
above it. A gap through the stop exits at the market (or, for a limit exit, waits at the
limit and says so).
"""

from datetime import datetime

from mocktrade.algos.engine import AlgoContext, BaseStrategy
from mocktrade.algos.execution import lots_text
from mocktrade.algos.spec import AlgoSpec, ParamKind, ParamSpec
from mocktrade.domain.orders import OrderType, Side
from mocktrade.domain.ticks import Tick
from mocktrade.orders.model import OrderEvent, OrderView, Pending

FAMILY = "protection"
HELD = "Held by MockTrade: works while the server runs."

EXIT = (
    ParamSpec(
        "then",
        "When touched",
        ParamKind.CHOICE,
        "market: exit at once at the best prices; limit: exit at a limit past the stop",
        default="market",
        choices=("market", "limit"),
    ),
    ParamSpec(
        "limit_ticks",
        "Limit slack",
        ParamKind.TICKS,
        "For a limit exit: how many ticks past the stop it may trade",
        default=5,
        min=0,
        max=500,
    ),
)


def touched(exit_side: Side, ltp: int, stop: int) -> bool:
    return ltp <= stop if exit_side is Side.SELL else ltp >= stop


def rupees(paise: int) -> str:
    return f"{paise / 100:,.2f}"


class _Stop(BaseStrategy):
    """The stop-and-exit part every protection algo shares."""

    def __init__(self, params: dict) -> None:
        self.market = params.get("then", "market") == "market"
        self.slack = params.get("limit_ticks", 5)
        self.stop: int | None = None
        self.triggered_at: int | None = None  # the last price that touched the stop

    def exit_price(self, ctx: AlgoContext, exit_side: Side) -> int | None:
        if self.market or self.stop is None:
            return None
        step = self.slack * ctx.instrument().tick_paise
        return self.stop - step if exit_side is Side.SELL else self.stop + step

    def send_exit(self, ctx: AlgoContext, qty: int, exit_side: Side, **kw) -> bool:
        price = self.exit_price(ctx, exit_side)
        kind = OrderType.MARKET if price is None else OrderType.LIMIT
        return ctx.send(qty, price, side=exit_side, order_type=kind, **kw)

    def exit_words(self, ctx: AlgoContext, exit_side: Side) -> str:
        price = self.exit_price(ctx, exit_side)
        return "at the market" if price is None else f"with a limit at {rupees(price)}"


# --- trailing stop ------------------------------------------------------------------------

TRAILING_SPEC = AlgoSpec(
    code="TRAILING_STOP",
    name="Trailing stop",
    help="A stop that follows the price by points or a percentage and never moves back. " + HELD,
    family=FAMILY,
    params=(
        ParamSpec(
            "trail_paise", "Trail (points)", ParamKind.PRICE, "Distance from the best price",
            required=False, min=1,
        ),
        ParamSpec(
            "trail_pct", "Trail (%)", ParamKind.PERCENT, "Or: distance as a percentage",
            required=False, min=0.05, max=20,
        ),
        *EXIT,
    ),
)  # fmt: skip


class TrailingStop(_Stop):
    """The parent's side is the exit side: SELL protects a long, BUY protects a short."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.points = params.get("trail_paise")
        self.pct = params.get("trail_pct")
        if (self.points is None) == (self.pct is None):
            raise ValueError("give the trail in points or in percent (one of them)")
        self.best: int | None = None  # the best price seen (high for a long, low for a short)

    def start(self, ctx: AlgoContext) -> None:
        tick = ctx.latest()
        if tick is None:
            ctx.say("waiting for a price")
            return
        self.on_tick(ctx, tick)

    def on_tick(self, ctx: AlgoContext, tick: Tick) -> None:
        if tick.instrument_id != ctx.instrument_id or self.triggered_at is not None:
            return
        ltp = tick.ltp_paise
        selling = ctx.side is Side.SELL  # protecting a long: follow the highs
        if self.best is None or (ltp > self.best if selling else ltp < self.best):
            self.best = ltp
            new = self._level(ctx, ltp)
            if self.stop is None or (new > self.stop if selling else new < self.stop):
                self.stop = new
        assert self.stop is not None
        ctx.report(stop_paise=self.stop, best_paise=self.best)
        if touched(ctx.side, ltp, self.stop):
            self.triggered_at = ltp
            self._exit(ctx)
        else:
            ctx.say(f"stop at {rupees(self.stop)}, {rupees(abs(ltp - self.stop))} away")

    def on_timer(self, ctx: AlgoContext, now: datetime) -> None:
        if self.triggered_at is not None:
            self._exit(ctx)  # a refused exit goes again (three refusals fail the algo)

    def _level(self, ctx: AlgoContext, price: int) -> int:
        distance = self.points if self.points is not None else price * self.pct / 100
        tick = ctx.instrument().tick_paise
        if ctx.side is Side.SELL:  # below the price, on the tick grid, never closer
            return int((price - distance) // tick * tick)
        return int(-((-(price + distance)) // tick) * tick)

    def _exit(self, ctx: AlgoContext) -> None:
        if ctx.working() or ctx.sending or ctx.held:
            return
        qty = ctx.remaining()
        if qty and self.send_exit(ctx, qty, ctx.side):
            assert self.stop is not None and self.triggered_at is not None
            ctx.say(
                f"stop {rupees(self.stop)} touched at {rupees(self.triggered_at)}: "
                f"exiting {qty} {self.exit_words(ctx, ctx.side)}"
            )


# --- OCO exit -----------------------------------------------------------------------------

OCO_SPEC = AlgoSpec(
    code="OCO",
    name="Target + stop (OCO)",
    help="Exits a position at a target or a stop, whichever comes first. " + HELD,
    family=FAMILY,
    params=(
        ParamSpec("target_paise", "Target", ParamKind.PRICE, "Take profit here", min=1),
        ParamSpec("stop_paise", "Stop", ParamKind.PRICE, "Cut the loss here", min=1),
        *EXIT,
    ),
)


class OcoExit(_Stop):
    """The parent's side is the exit side; its quantity is what to exit."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.target = params["target_paise"]
        self.stop = params["stop_paise"]

    def start(self, ctx: AlgoContext) -> None:
        selling = ctx.side is Side.SELL
        assert self.stop is not None
        if (self.target <= self.stop) if selling else (self.target >= self.stop):
            where = "above" if selling else "below"
            return ctx.fail(f"the target must be {where} the stop for a {ctx.side.value} exit")
        tick = ctx.latest()
        if tick is not None and touched(ctx.side, tick.ltp_paise, self.stop):
            return ctx.fail(f"the last price {rupees(tick.ltp_paise)} is already past the stop")
        ctx.report(target_paise=self.target, stop_paise=self.stop)
        if ctx.send(ctx.qty, self.target):
            ctx.say(f"target {rupees(self.target)} working; stop {rupees(self.stop)} watched")

    def on_tick(self, ctx: AlgoContext, tick: Tick) -> None:
        assert self.stop is not None
        if self.triggered_at is None and touched(ctx.side, tick.ltp_paise, self.stop):
            self.triggered_at = tick.ltp_paise
            ctx.say(f"stop touched at {rupees(tick.ltp_paise)}: cancelling the target")
            self._follow_up(ctx)

    def on_child(self, ctx: AlgoContext, child: OrderView, event: OrderEvent) -> None:
        self._follow_up(ctx)

    def on_timer(self, ctx: AlgoContext, now: datetime) -> None:
        self._follow_up(ctx)

    def _follow_up(self, ctx: AlgoContext) -> None:
        """After the stop: first the target goes, then what's still open is exited."""
        if self.triggered_at is None:
            return
        targets = [c for c in ctx.working() if c.request.price_paise == self.target]
        for c in targets:
            if c.pending is Pending.NONE:
                ctx.cancel(c.client_order_id)
        if targets or ctx.working() or ctx.sending or ctx.held:
            return  # the target's cancel isn't confirmed yet, or the exit is working
        qty = ctx.remaining()
        if qty and self.send_exit(ctx, qty, ctx.side):
            ctx.say(f"stop touched: exiting {qty} {self.exit_words(ctx, ctx.side)}")


# --- bracket ------------------------------------------------------------------------------

BRACKET_SPEC = AlgoSpec(
    code="BRACKET",
    name="Bracket",
    help="An entry with a target and a stop placed as it fills; the first one hit closes "
    "the trade. " + HELD,
    family=FAMILY,
    params=(
        ParamSpec(
            "entry_paise", "Entry", ParamKind.PRICE, "Limit price to enter (empty: market)",
            required=False, min=1,
        ),
        ParamSpec(
            "target_points_paise", "Target (points)", ParamKind.PRICE,
            "Profit distance from the entry price", min=1,
        ),
        ParamSpec(
            "stop_points_paise", "Stop (points)", ParamKind.PRICE,
            "Loss distance from the entry price", min=1,
        ),
        *EXIT,
    ),
)  # fmt: skip


class Bracket(_Stop):
    """The parent's side and quantity are the entry; exits are on the other side."""

    def __init__(self, params: dict) -> None:
        super().__init__(params)
        self.entry = params.get("entry_paise")
        self.target_points = params["target_points_paise"]
        self.stop_points = params["stop_points_paise"]
        self.target: int | None = None  # fixed at the first entry fill's price
        self.exit_sending = 0  # exit quantity sent but not seen as a child yet
        self.seen: set[str] = set()

    def start(self, ctx: AlgoContext) -> None:
        kind = OrderType.MARKET if self.entry is None else OrderType.LIMIT
        if ctx.send(ctx.qty, self.entry, order_type=kind):
            at = "at the market" if self.entry is None else f"at {rupees(self.entry)}"
            ctx.say(f"entry {ctx.side.value.lower()} {ctx.qty} {at}")

    def exit_side(self, ctx: AlgoContext) -> Side:
        return Side.SELL if ctx.side is Side.BUY else Side.BUY

    def on_child(self, ctx: AlgoContext, child: OrderView, event: OrderEvent) -> None:
        if child.request.side is not ctx.side and child.client_order_id not in self.seen:
            self.seen.add(child.client_order_id)
            self.exit_sending = max(0, self.exit_sending - child.request.qty)
        self._sync(ctx)

    def on_tick(self, ctx: AlgoContext, tick: Tick) -> None:
        if self.stop is None or self.triggered_at is not None:
            return
        if touched(self.exit_side(ctx), tick.ltp_paise, self.stop):
            self.triggered_at = tick.ltp_paise
            ctx.say(f"stop {rupees(self.stop)} touched at {rupees(tick.ltp_paise)}")
            self._sync(ctx)

    def on_timer(self, ctx: AlgoContext, now: datetime) -> None:
        self._sync(ctx)

    def done(self, ctx: AlgoContext) -> bool:
        entered = ctx.filled(ctx.instrument_id, ctx.side)
        exited = ctx.filled(ctx.instrument_id, self.exit_side(ctx))
        entry_over = entered >= ctx.qty or (
            self.triggered_at is not None and not self._working(ctx, ctx.side)
        )
        return entered > 0 and entry_over and exited >= entered and not ctx.working()

    def on_finish(self, ctx: AlgoContext) -> str | None:
        open_qty = self._open(ctx)
        if open_qty > 0:
            return f"UNPROTECTED: {open_qty} of {ctx.instrument_id} is open with no stop or target"
        return None

    # --- inside ---

    def _working(self, ctx: AlgoContext, side: Side) -> list[OrderView]:
        return [c for c in ctx.working() if c.request.side is side]

    def _open(self, ctx: AlgoContext) -> int:
        entered = ctx.filled(ctx.instrument_id, ctx.side)
        return entered - ctx.filled(ctx.instrument_id, self.exit_side(ctx))

    def _sync(self, ctx: AlgoContext) -> None:
        exit_side = self.exit_side(ctx)
        entered = ctx.filled(ctx.instrument_id, ctx.side)
        if entered and self.target is None:
            self._set_levels(ctx)
        open_qty = self._open(ctx)
        exits = self._working(ctx, exit_side)
        covered = sum(c.leaves_qty for c in exits) + self.exit_sending
        ctx.report(open_qty=open_qty)
        if ctx.held or self.target is None:
            return
        if self.triggered_at is None:
            if open_qty - covered > 0:
                qty = open_qty - covered
                if ctx.send(qty, self.target, side=exit_side, check_remaining=False):
                    self.exit_sending += qty
                    ctx.say(
                        f"{lots_text(entered // ctx.instrument().lot_size)} in; target "
                        f"{rupees(self.target)} working, stop {rupees(self.stop or 0)} watched"
                    )
            return
        # the stop was touched: first the entry's rest and the targets go, then exit
        others = self._working(ctx, ctx.side) + [
            c for c in exits if c.request.price_paise == self.target
        ]
        for c in others:
            if c.pending is Pending.NONE:
                ctx.cancel(c.client_order_id)
        if others or exits or self.exit_sending:
            return
        if open_qty > 0 and self.send_exit(ctx, open_qty, exit_side, check_remaining=False):
            self.exit_sending += open_qty
            ctx.say(f"stop touched: exiting {open_qty} {self.exit_words(ctx, exit_side)}")

    def _set_levels(self, ctx: AlgoContext) -> None:
        price = ctx.average_price()
        assert price is not None
        up = ctx.side is Side.BUY
        tick = ctx.instrument().tick_paise
        down = lambda x: x // tick * tick  # noqa: E731
        upward = lambda x: -(-x // tick) * tick  # noqa: E731
        # on the tick grid, rounded toward the entry: a target easier to reach, a stop tighter
        target = price + self.target_points if up else price - self.target_points
        stop = price - self.stop_points if up else price + self.stop_points
        self.target = down(target) if up else upward(target)
        self.stop = upward(stop) if up else down(stop)
        ctx.report(target_paise=self.target, stop_paise=self.stop, entry_avg_paise=price)
