class BaseAgent:

  def __init__(self, capital):
    self.capital = capital
    self.engine = None

  def register_engine(self, engine):
    self.engine = engine

  @property
  def data(self):
    return self.engine.data
