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
    
    @classmethod
    def from_str(cls, string):
        first, last = string.find(":"), string[::-1].find(":")
        return PoolName(string[:first], string[first+1:-last-1], string[-last:])
    
    def __eq__(self, other):
        if isinstance(other, PoolName):
            return (self.chain, self.protocol, self.token_name) == (other.chain, other.protocol, other.token_name)
        elif isinstance(other, str):
            return repr(self) == other
        return False
    
    def to_disk_name(self):
        return f"chain___{self.chain}___protocol___{self.protocol}___token___{self.token_name}"
    
    @classmethod
    def try_from_disk_name(cls, disk_name, verbose=True):
        try:
            name_lst = disk_name.split("___")
            return cls(chain=name_lst[1], protocol=name_lst[3], token_name=name_lst[5])
        except Exception as e:
            if verbose:
                print(f"Warning: {e} when loading {disk_name}.")
            return disk_name
    