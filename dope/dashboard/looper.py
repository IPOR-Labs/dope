from functools import lru_cache
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets
from IPython.display import display

from dope.dashboard import looper_cfg as plot_cfg
from dope.dashboard.looper_model import LoopModel


def make_linkable_cb(slides, obj, link_name):
      def on_checkbox_change(change):
          if change['new']:  # If the checkbox is checked
            # Link the sliders
            obj.__setattr__(link_name, widgets.jslink(*[(s,"value") for s in slides ]))
          else:
              # Unlink the sliders if they are linked
              link = obj.__getattribute__(link_name)
              if link is not None:
                  link.unlink()
                  obj.__setattr__(link_name, None)
      return on_checkbox_change



class LoopDashboard:
  def __init__(self, maestro, init_params=None):
    
    self.model = LoopModel()
    
    self.dfs, self.level_caps = self.model.get_df_and_caps(
        pool_dict=maestro.pools,
        rates_data_collection=maestro.rates_data_collection,
        price_data_collection=maestro.price_data_collection,
    )
    self.loops = list(self.dfs.keys())
    
    self.params = init_params or dict(
        loop=list(self.dfs.keys())[0],
        df=list(self.dfs.values())[0],
        mavg=7,
    )
    self.params["leverage"] = self.level_caps[self.params["loop"]]
    self.cfg = plot_cfg.Config()
    self.plot_config = plot_cfg.get_plot_cfg()
  
  @property
  def model_params(self):
    params = {k:v for k,v in self.params.items() if k not in ["data"]}
    
    return {
      "meta":{"contract": self.contracts[self.timeseries_tabs.selected_index]},
      "params": params, 
      "leverage_params": self.leverage_params,
      }
    
  
  def get_timeseries_fig(self, title="Looping APY Timeseries"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, # subplot_titles=(title, "Leverage"),
                    row_heights=[0.7, 0.3])
    fig = go.FigureWidget(fig)  # Ensure it's a FigureWidget for dynamic updates
    fig.update_layout(title_text=title, height=500, width=950)
    #fig.update_yaxes(range=[0, int(self.leverage_params["cap"]*1.1)], row=2, col=1)
    # y label:
    fig.update_yaxes(title_text="APY", row=1, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig

  def update_timeseries(self, fig, df):
    token = str(self.params["loop"])
    fig.update_layout(title_text=f"{token.upper()}")
    with fig.batch_update():
      ploted = []
      for data in fig.data:
        cfg = self.plot_config[data.name]
        if "should_show_col" in cfg:
          should_plot = self.cfg.__getattribute__(cfg["should_show_col"])
        else: 
          should_plot = True
        if should_plot:
          df_col = cfg["df_col"]
          if ("accipor" in data.name):
            df_col = cfg["df_col"]
          data.y  = df[df_col]
          ploted.append(data.name)
        else:
          data.y = []
          ploted.append(data.name)

      # if data is available, but needs to be ploted from zero
      for data_name, cfg in self.plot_config.items():
        if data_name not in ploted:
          if "should_show_col" in cfg:
            should_plot = self.cfg.__getattribute__(cfg["should_show_col"])
          else: 
            should_plot = True
          if should_plot:
            df_col = cfg["df_col"]
            if ("accipor" in data_name):
              df_col = cfg["df_col"]
            fig.add_trace(go.Scatter(x=df.index, y=df[df_col], **cfg["scatter_cfg"]), row=cfg["row"], col=cfg["col"])

  def on_dropdown_change(self, change):
    if change['type'] == 'change' and change['name'] == 'value':
        loop = change['new']
        self.params["loop"] = loop
        self.params["df"] = self.dfs[loop]

  def on_slide_change(self, change):
    _slider = change["owner"]
    if hasattr(_slider, "widget_name"):
      if _slider.widget_name in self.params:
        self.params[_slider.widget_name] = change["new"] / _slider.divisor
      else:
        print("Unknown slider", _slider.widget_name)


  def on_change(self, return_raw=False):
      self.is_running_label.value = "Running..."
      self.df = self.model(**self.params)
      if return_raw:
        self.is_running_label.value = "Done"
        return self.df
      self.df = self.df.resample("1D").last()

      self.update_timeseries(self.timeseries_fig, self.df) 
      self.is_running_label.value = "Done"


  def get_k_spread(self, title, name, value=1.5, _min=0, _max=3, step=0.05, orientation="vertical"):
    _slide = widgets.FloatSlider(
        value=value,
        min=_min,
        max=_max,
        step=step,
        description=title,
        disabled=False,
        continuous_update=False,
        orientation=orientation,
        readout=True,
        #readout_format='dd'
    )
    _slide.widget_name = name
    _slide.divisor = 1
    return _slide


  def get_mvag_slider(self, title, name, _min=1, _max=30, step=1, orientation="horizontal"):

    _slide = widgets.IntSlider(
        value=self.params[name],
        min=_min,
        max=_max,
        step=step,
        description=title,
        disabled=False,
        continuous_update=False,
        orientation=orientation,
        readout=True,
        readout_format='d'
    )
    _slide.widget_name = name
    _slide.divisor = 1
    return _slide


  def get_spread_slider(self, title, name, value=10):
    _slide = widgets.IntSlider(
        value=10,
        min=0,
        max=1500,
        step=1,
        description=title,
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        readout_format='d'
    )
    _slide.widget_name = name
    _slide.divisor = 10_000
    return _slide

  def get_config_checkbox(self, title, name):
    _cb = widgets.Checkbox(
        value=True,
        description=title,
        disabled=False,
        indent=False
    )
    _cb.widget_name = name
    return _cb

  def get_config_show_buttons(self):
    options = list(self.cfg.__dataclass_fields__.keys())
    
    def on_config_button_click(b):  
        #print(b.description, self.cfg.__getattribute__(b.description), end=" >>> ")
        self.cfg.__setattr__(b.description, not self.cfg.__getattribute__(b.description))
        b.button_style = 'info' if self.cfg.__getattribute__(b.description) else ''
        #print(b.description, self.cfg.__getattribute__(b.description))

    buttons = []
    for option in options:
      _style = 'info' if self.cfg.__getattribute__(option) else ''
      b = widgets.Button(description=option, button_style=_style)  
      buttons.append(b)
    
    buttons_per_row = 3 # +1 

    self.config_button_rows = []
    button_row = []
    i = 0
    _rows = []
    for button in buttons:
      button.on_click(on_config_button_click)
      button_row.append(button)
      i += 1
      if i > buttons_per_row:
        i = 0
        _rows.append(widgets.HBox(button_row))
        button_row = []
    _rows.append(widgets.HBox(button_row))
  
    return widgets.VBox(_rows)
      
  def get_mu_vol_spread_component(self):
    # mu_spread_factor_pay
    _max, step = 1, 0.01
    vol_label = widgets.Label(value="Vol Band Spread")
    vol_spread_slides = [
      self.get_k_spread(
        "$f_{\\sigma}$ Pay", 
        "vol_spread_factor_pay", 
        value=self.params["vol_spread_factor_pay"], _max=3, step=step),
      self.get_k_spread("$f_{\\sigma}$ Rec", "vol_spread_factor_rec", value=self.params["vol_spread_factor_rec"], _max=3, step=step),
      ]
    
    checkbox_vol = widgets.Checkbox(description='Link $f_{\\sigma}$', value=True,
                                    layout=widgets.Layout(width='170px', align_items='center')) 
    self._link_vol = None
    cb_vol = make_linkable_cb(vol_spread_slides, self, link_name="_link_vol")
    checkbox_vol.observe(cb_vol, names='value')
    cb_vol({"new":False})


    mu_label = widgets.Label(value="$\\mu$ Band Spread")
    mu_spread_slides = [
      self.get_k_spread("$f_{\\mu}$ Pay", "mu_spread_factor_pay", value=self.params["mu_spread_factor_pay"], _max=_max, step=step),
      self.get_k_spread("$f_{\\mu}$ Rec", "mu_spread_factor_rec", value=self.params["mu_spread_factor_rec"], _max=_max, step=step),
      ]

    checkbox_mu = widgets.Checkbox(description='Link $f_{\\mu}$', value=True,
                                   layout=widgets.Layout(width='170px', align_items='center')) 
    self._link_mu = None
    cb_mu = make_linkable_cb(mu_spread_slides, self, link_name="_link_mu")
    checkbox_mu.observe(cb_mu, names='value')
    cb_mu({"new":False})
    
    vbox_vol_bands = widgets.VBox(
      [vol_label, widgets.HBox(vol_spread_slides), checkbox_vol], 
      layout=widgets.Layout(align_items='center')
    )
    
    vbox_mu_bands = widgets.VBox(
      [mu_label, widgets.HBox(mu_spread_slides), checkbox_mu], 
      layout=widgets.Layout(align_items='center')
    )
    for slide in vol_spread_slides + mu_spread_slides:
      slide.observe(self.on_slide_change, names="value")
      
    return widgets.HBox([vbox_vol_bands, vbox_mu_bands]) 

  def setup(self):

    label = widgets.Label(value="Moving Avg. Window (days)")
    mavg_sliders = [
      self.get_k_spread("Leverage", "leverage", value=self.params["leverage"], _max=self.level_caps[self.params["loop"]], step=0.05, orientation="horizontal" ),
      self.get_mvag_slider("Moving Avg.", "mavg"),  
      ]
    
    # mavg_sliders[0].observe(lambda _: self.on_change(), names='value')
    self.vbox_mavg_sliders = widgets.VBox([label, widgets.VBox(mavg_sliders)], layout=widgets.Layout(align_items='center'))

    self.timeseries_fig = self.get_timeseries_fig()    
    self.timeseries_tabs = self.timeseries_fig 

    self.token_dropdown = widgets.Dropdown(
        options=self.loops,
        value=self.loops[0],
        description='Loop:',
        disabled=False,
    )

    self.token_dropdown.observe(self.on_dropdown_change, names='value')
    for slide in mavg_sliders:
      slide.observe(self.on_slide_change, names="value")
  
  
    # config_button_column = self.get_config_show_buttons()
    
    # self.stress_test_scenario_column = self.stress_test_scenario()
    self.header_bottom = widgets.HBox([self.vbox_mavg_sliders, ])
    
    self.run_button = widgets.Button(description="Run")
    self.run_button.on_click(lambda _: self.on_change())
    self.run_button.button_style = 'primary'
    
    # self.run_button = widgets.Button(description="Run")
    # self.run_button.on_click(lambda _: self.on_change())
    # self.run_button.button_style = 'primary'
    
    self.is_running_label = widgets.Label(value="Done")

    self.header = widgets.HBox([self.token_dropdown, self.run_button, self.is_running_label])
  

  def register_new_params(self, params):
    def walktree_change_values(section):
      children = getattr(section, 'children', None)
      if children is None:
        return
      for child in children:
        widget_name = getattr(child, 'widget_name', None)
        if widget_name is not None:
          if widget_name in params:
            child.value = params[widget_name] * child.divisor
            self.params[widget_name] = params[widget_name]
          pass # change value
        walktree_change_values(child)
        
    # walktree_change_values(self.spread_section)
    walktree_change_values(self.header_bottom)
  
  def suggest_params(self, start_on_zero=False, trials=100):
    from scontrol.dashboard.optimizer import ModelOptimizer
    contract = self.contracts[self.timeseries_tabs.selected_index]
    self.optimizer = ModelOptimizer(self.model, self.params, fixed_keys=["token", "data"], maturity=contract, side="pay")
    
    if start_on_zero:
      initial_guess = self.optimizer.get_a_zero_guess()
    else:
      initial_guess = self.model_params.copy()
    
    initial_guess["token"] = self.params["token"]
    initial_guess["data"] = self.params["token"]
    
    best_params = self.optimizer.optimize(n_trials=100, initial_guess=initial_guess, verbose=False)
    
    self.register_new_params(best_params)
    return best_params
      
    

      
  def display(self):
    display(self.header, self.header_bottom, self.timeseries_tabs)
    #display(tabs)
    self.on_change()