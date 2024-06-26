class BasePredictor:

  def __init__(self):
    pass

  def register_agent(self, agent):
    self.agent = agent

  @property
  def data_ref(self):
    return self.agent.data
