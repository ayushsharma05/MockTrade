"""The algos MockTrade ships with. Add one here (spec + strategy) and every screen shows it."""

from mocktrade.algos import iceberg, pov, protect, schedule, slicer, spreader
from mocktrade.algos.engine import AlgoType

BUILTIN: tuple[AlgoType, ...] = (
    AlgoType(slicer.SPEC, slicer.Slicer),
    AlgoType(schedule.TWAP_SPEC, schedule.Twap),
    AlgoType(schedule.VWAP_SPEC, schedule.Vwap),
    AlgoType(pov.SPEC, pov.Pov),
    AlgoType(iceberg.SPEC, iceberg.Iceberg),
    AlgoType(spreader.SPEC, spreader.Spreader),
    AlgoType(protect.TRAILING_SPEC, protect.TrailingStop),
    AlgoType(protect.OCO_SPEC, protect.OcoExit),
    AlgoType(protect.BRACKET_SPEC, protect.Bracket),
)
