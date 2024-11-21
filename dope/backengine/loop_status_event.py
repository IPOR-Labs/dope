from dataclasses import dataclass
from typing import Dict

@dataclass
class StatusEvent:
    datetime: str
    weights: Dict
    rate: Dict
    health_factor: Dict
    capital: float
    capital_breakdown: Dict
    token_breakdown: Dict
    r_breakdown: Dict
    r_by_weight: Dict
    impact_breakdown: Dict
    prices: Dict
    
    def as_row(self):
        event_row = []
        for key in self.__dataclass_fields__.keys():
            event_row.append(getattr(self, key))
        return event_row

    @classmethod
    def columns(cls):
        return list(cls.__dataclass_fields__.keys())