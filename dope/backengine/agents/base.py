class BaseAgent:

    def __init__(self, capital):
        self.capital = capital
        self.engine = None

    def register_engine(self, engine):
        self.engine = engine

    @property
    def data(self):
        return self.engine.borrow_lend_data
    
    def on_start(self, *args, **kwargs):
        pass
    
    def on_act(self, *args, **kwargs):
        pass
