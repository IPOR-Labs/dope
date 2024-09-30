from dataclasses import dataclass

@dataclass
class Token:
    value: float
    name: str

    # Addition with another Token or scalar
    def __add__(self, other):
        if isinstance(other, Token):
            if self.name != other.name:
                raise ValueError(f"Cannot add tokens with different names: {self.name} != {other.name}")
            return Token(self.value + other.value, self.name)
        raise ValueError(f"Cannot add Token and {type(other)}")

    # In-place addition with another Token or scalar (for +=)
    def __iadd__(self, other):
        if isinstance(other, Token):
            if self.name != other.name:
                raise ValueError(f"Cannot add tokens with different names: {self.name} != {other.name}")
            self.value += other.value
            return self
        raise ValueError(f"Cannot add Token and {type(other)}")
        

    # Subtraction with another Token or scalar
    def __sub__(self, other):
        if isinstance(other, Token):
            if self.name != other.name:
                raise ValueError(f"Cannot subtract tokens with different names: {self.name} != {other.name}")
            return Token(self.value - other.value, self.name)
        raise ValueError(f"Cannot subtract Token and {type(other)}")

    # In-place subtraction with another Token or scalar (for -=)
    def __isub__(self, other):
        if isinstance(other, Token):
            if self.name != other.name:
                raise ValueError(f"Cannot subtract tokens with different names: {self.name} != {other.name}")
            self.value -= other.value
            return self
        raise ValueError(f"Cannot subtract Token and {type(other)}")

    # Multiplication with a scalar (int or float)
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Token(self.value * other, self.name)
        return NotImplemented

    # In-place multiplication with scalar (for *=)
    def __imul__(self, other):
        if isinstance(other, (int, float)):
            self.value *= other
            return self
        return NotImplemented

    # Division with a scalar (int or float)
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Token(self.value / other, self.name)
        return NotImplemented

    # In-place division with scalar (for /=)
    def __itruediv__(self, other):
        if isinstance(other, (int, float)):
            self.value /= other
            return self
        return NotImplemented

    # Equality check
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.name == other.name and self.value == other.value
        return False

    # Representation for convenience
    def __repr__(self):
        return f"Token({self.value}, '{self.name}')"
