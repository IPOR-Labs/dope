from dataclasses import dataclass
from typing import Any

@dataclass
class PoolName:
    chain: str
    protocol: str
    token_name:str
    
    def __repr__(self) -> str:
        return f"{self.chain}:{self.protocol}:{self.token_name}"
    
    def __hash__(self) -> int:
        return hash(repr(self))
    
    def __getitem__(self, key):
        return getattr(self, key)
    
    def __iter__(self):
        for attr in dir(self):
            if not attr.startswith("__") and not callable(getattr(self, attr)):
                yield getattr(self, attr)
    
    def to_disk_name(self):
        return f"chain___{self.chain}___protocol___{self.protocol}___token___{self.token_name}"
    
    @classmethod
    def from_disk_name(cls, disk_name):
        disk_name = disk_name.split("___")
        return cls(chain=disk_name[1], protocol=disk_name[3], token_name=disk_name[5])
    