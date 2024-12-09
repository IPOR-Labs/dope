from functools import lru_cache
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import ipywidgets as widgets
from IPython.display import display


from dope.dashboard.looper_cfg import plot_cfg


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


class QuotesDashboard:
  def __init__(self, dfs, model, init_params=None, leverage_params=None):
    self.dfs = dfs
    self.model = model
    
    self.params = init_params or dict(token = "dai",
      data = dfs["dai"],
      alpha_low_pay  = 0.0010,
      alpha_low_rec  = 0.0010,

      alpha_high_pay = 0.0010,
      alpha_high_rec = 0.0010,

      k_upper = 1.5,
      k_lower = 1.5,
      low_vol_mavg_window_days=7,
      high_vol_mavg_window_days=28,
      dynamic_cap_window_days=28,
    )
    self.leverage_params = leverage_params or dict(cap=500)
    self.cfg = plot_cfg.Config()
    self.plot_config = plot_cfg.get_plot_cfg()
    self.hw = HullWhiteModel()
  
  @property
  def model_params(self):
    params = {k:v for k,v in self.params.items() if k not in ["data"]}
    
    return {
      "meta":{"contract": self.contracts[self.timeseries_tabs.selected_index]},
      "params": params, 
      "leverage_params": self.leverage_params,
      }
    
  
  def get_timeseries_fig(self, title="IPOR Quote over time"):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    vertical_spacing=0.05, subplot_titles=(title, "Leverage"),
                    row_heights=[0.85, 0.15])  # Adjust row heights as needed)
    fig = go.FigureWidget(fig)  # Ensure it's a FigureWidget for dynamic updates
    fig.update_layout(title_text=title, height=600, width=950)
    fig.update_yaxes(range=[0, int(self.leverage_params["cap"]*1.1)], row=2, col=1)

    return fig

  def update_timeseries(self, fig, df):
    # fig.data = []  # Clear existing data
    #fig.layout.shapes = []

    contract = self.contracts[self.timeseries_tabs.selected_index]
    token = self.params["token"]
    fig.update_layout(title_text=f"{token.upper()} : {contract} Days")
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
          if ("accipor" in data.name) or ("leverage" in data.name):
            contract = self.contracts[self.timeseries_tabs.selected_index]
            df_col = cfg["df_col"] + f"_{contract}"
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
            if ("accipor" in data_name) or ("leverage" in data_name):
              contract = self.contracts[self.timeseries_tabs.selected_index]
              df_col = cfg["df_col"] + f"_{contract}"
            fig.add_trace(go.Scatter(x=df.index, y=df[df_col], **cfg["scatter_cfg"]), row=cfg["row"], col=cfg["col"])

  # Function to handle dropdown changes
  def on_dropdown_change(self, change):
    if change['type'] == 'change' and change['name'] == 'value':
      token = change['new']
      #print(token)
      self.params["token"] = token
      self.params["data"] = self.dfs[token]
      #self.on_change()

  def on_slide_change(self, change):
    _slider = change["owner"]
    if hasattr(_slider, "widget_name"):
      #print(change["new"], _slider.widget_name)
      if _slider.widget_name in self.params:
        self.params[_slider.widget_name] = change["new"] / _slider.divisor
      elif _slider.widget_name in self.leverage_params:
        self.leverage_params[_slider.widget_name] = change["new"] / _slider.divisor
      else:
        print("Unknown slider", _slider.widget_name)
      
      #self.on_change()

  def on_change(self, return_raw=False):
      self.is_running_label.value = "Running..."
      self.df = self.model.historical_backtest(**self.params)
      self.df = self.model.historical_leverage(self.df, **self.leverage_params)
      if return_raw:
        self.is_running_label.value = "Done"
        return self.df
      self.df = self.df.resample("1D").last()
      ix = self.timeseries_tabs.selected_index
      self.update_timeseries(self.timeseries_fig[self.contracts[ix]], self.df) 
      self.is_running_label.value = "Done"

  # Example of a slider widget

  def get_k_spread(self, title, name, value=1.5, _min=0, _max=3, step=0.05):
    _slide = widgets.FloatSlider(
        value=value,
        min=_min,
        max=_max,
        step=step,
        description=title,
        disabled=False,
        continuous_update=False,
        orientation='vertical',
        readout=True,
        #readout_format='dd'
    )
    _slide.widget_name = name
    _slide.divisor = 1
    return _slide


  def get_mvag_slider(self, title, name):

    _slide = widgets.IntSlider(
        value=self.params[name],
        min=0,
        max=90,
        step=1,
        description=title,
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
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
  
  @lru_cache(maxsize=None)
  def _calibrate_hull_white(self, token):
    if token == "dai":
      return 0.11605318682790618, 0.043041039162823265, 0.05150443915488017
    if token == "lido":
      return 0.02509545933026399, 0.049700942546430656, 0.03439537380842012
    print("Calibrating Interest Rate Model. It should take 2 minutes...")
    self.hw.calibrate(self.dfs[token].r_t)
    print("Calibration Complete", self.hw.a, self.hw.sigma, self.hw.theta)
    return self.hw.a, self.hw.sigma, self.hw.theta

  def generate_synthetic_data(self):
    if "is_real_data" in self.params["data"]:
      self.params["data"] = self.params["data"][self.params["data"].is_real_data]

    print("Generating Synthetic Data", self.params["token"])
    a, sigma, theta = self._calibrate_hull_white(self.params["token"])
    vol_factor = self.added_vol.value
    added_mean = self.new_long_term_mean.value
    hw = HullWhiteModel(a, sigma*vol_factor, theta+added_mean)

    r0 = self.params["data"].r_t.iloc[-1]
    start_date = self.params["data"].index[-1] + pd.DateOffset(days=1)
    end_date = start_date + pd.DateOffset(days=100)

    self.params["data"]["is_real_data"] = True
    print(r0, start_date, end_date)
    syn_rt = hw.simulate(r0, start_date, end_date)
    syn_data = pd.DataFrame()
    syn_data["r_t"] = syn_rt
    syn_data["is_real_data"] = False

    extra_data = pd.concat([self.params["data"], syn_data], axis=0).reset_index().rename(columns={"index":"datetime"})
    T = 60 * 60 * 24 * 365
    extra_data["dt"] = extra_data.datetime.diff().apply(lambda d: d.total_seconds()/T).shift(-1).values
    
    from scontrol.transformers.builder import Builder
    mock_builder = Builder(self.params["token"], date_from=None, date_to=None, load_data=False, agg=None)
    contract = self.contracts[self.timeseries_tabs.selected_index]
    extra_data = mock_builder.add_accrued_ipor(extra_data, contract_length=contract)
    extra_data.set_index("datetime", inplace=True)
    self.params["data"] = extra_data

    #self.on_change()
  
  def stress_test_scenario(self):
    label = widgets.Label(value="Stress Test Scenario:")
    self.simulate_button = widgets.Button(description="Simulate", button_style="primary")
    self.remove = widgets.Button(description="Remove", button_style="warning")

    self.simulate_button.on_click(lambda _: self.generate_synthetic_data())
    def on_remove_click(_):
      self.params["data"] = self.dfs[self.params["token"]]
      self.on_change()
    self.remove.on_click(on_remove_click)

    self.added_vol = widgets.FloatText(
      value=1,  # Initial value
      placeholder=1,
      description='(x) Vol Factor:',
      disabled=False
    )
    self.new_long_term_mean = widgets.FloatText(
      value=0.05,  # Initial value
      placeholder=0.05,
      description='(+) Mean Factor:',
      disabled=False
    )
    
    # new_long_term_mean = widgets.Text(
    #   value=0,  # Initial value
    #   placeholder='0',
    #   description='Long Term Mean:',
    #   disabled=False
    # )
    
    return widgets.VBox([
      label, 
      widgets.VBox([self.simulate_button, self.remove]),
      widgets.VBox([self.added_vol, self.new_long_term_mean]), 
      
    ])
    
  def get_mu_vol_spread_component(self):
    # mu_spread_factor_pay
    _max, step = 1, 0.01
    vol_label = widgets.Label(value="Vol Band Spread")
    vol_spread_slides = [
      self.get_k_spread(
        "$f_{\sigma}$ Pay", 
        "vol_spread_factor_pay", 
        value=self.params["vol_spread_factor_pay"], _max=3, step=step),
      self.get_k_spread("$f_{\sigma}$ Rec", "vol_spread_factor_rec", value=self.params["vol_spread_factor_rec"], _max=3, step=step),
      ]
    
    checkbox_vol = widgets.Checkbox(description='Link $f_{\sigma}$', value=True,
                                    layout=widgets.Layout(width='170px', align_items='center')) 
    self._link_vol = None
    cb_vol = make_linkable_cb(vol_spread_slides, self, link_name="_link_vol")
    checkbox_vol.observe(cb_vol, names='value')
    cb_vol({"new":False})


    mu_label = widgets.Label(value="$\mu$ Band Spread")
    mu_spread_slides = [
      self.get_k_spread("$f_{\mu}$ Pay", "mu_spread_factor_pay", value=self.params["mu_spread_factor_pay"], _max=_max, step=step),
      self.get_k_spread("$f_{\mu}$ Rec", "mu_spread_factor_rec", value=self.params["mu_spread_factor_rec"], _max=_max, step=step),
      ]

    checkbox_mu = widgets.Checkbox(description='Link $f_{\mu}$', value=True,
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

    label = widgets.Label(value="Low Vol Spreads")
    low_vol_spread_slides = [
      self.get_spread_slider("Pay", "alpha_low_pay", value=self.params["alpha_low_pay"]),
      self.get_spread_slider("Rec", "alpha_low_rec", value=self.params["alpha_low_rec"]),
    ]
    vbox_high = widgets.VBox([label, widgets.HBox(low_vol_spread_slides)], layout=widgets.Layout(align_items='center'))

    label = widgets.Label(value="High Vol Spreads")
    high_vol_spread_slides = [
      self.get_spread_slider("Pay", "alpha_high_pay", value=self.params["alpha_high_pay"]),
      self.get_spread_slider("Rec", "alpha_high_rec", value=self.params["alpha_high_rec"]),
    ]
    # print(f'{self.params["alpha_high_pay"] = }')
    # for slide in low_vol_spread_slides + high_vol_spread_slides:
    #   print(slide.value, slide)
    vbox_low = widgets.VBox([label, widgets.HBox(high_vol_spread_slides)], layout=widgets.Layout(align_items='center'))

    label = widgets.Label(value="Vol Bands ($k\cdot \sigma$)")
    vol_bands = [
      self.get_k_spread("K Upper", "k_upper", value=self.params["k_upper"]),
      self.get_k_spread("K Lower", "k_lower", value=self.params["k_lower"]),
    ]
    vbox_bands = widgets.VBox([label, widgets.HBox(vol_bands)], layout=widgets.Layout(align_items='center'))
    
    
    label = widgets.Label(value="Leverage")
    level_bands = [
      self.get_k_spread("Vol Factor (K)", "vol_factor", value=self.leverage_params["vol_factor"], _max=10, step=0.01, _min=0),
    ]
    leverage_bands = widgets.VBox([label, widgets.HBox(level_bands)], layout=widgets.Layout(align_items='center'))
    

    vbox_vol_bands = self.get_mu_vol_spread_component()

    label = widgets.Label(value="Moving Avg. Window (days)")
    mavg_sliders = [
      self.get_mvag_slider("Low Vol MAvg", "low_vol_mavg_window_days"),
      self.get_mvag_slider("High Vol MAvg", "high_vol_mavg_window_days"),
      self.get_mvag_slider("Dynamic Cap", "dynamic_cap_window_days"),
      self.get_mvag_slider("Vol Band Spread:", "vol_spread_window_days"),
      
      ]
    
    self.vbox_mavg_sliders = widgets.VBox([label, widgets.VBox(mavg_sliders)], layout=widgets.Layout(align_items='center'))

    self.spread_section = widgets.HBox([vbox_high, vbox_low, vbox_bands, vbox_vol_bands, leverage_bands])
    self.timeseries_fig = {
      4*7: self.get_timeseries_fig(),
      7*8: self.get_timeseries_fig(),
      7*12: self.get_timeseries_fig()
    }
    self.contracts = list(self.timeseries_fig.keys())
    
    # self.leverages = {k:self.get_timeseries_fig(title="Leverage") for k in self.contracts}
    
    self.timeseries_tabs = widgets.Tab(children=[widgets.VBox([self.timeseries_fig[c]]) for c in self.contracts])
    i = 0
    for k in self.contracts:
      self.timeseries_tabs.set_title(i, f"Contract {k}D")
      i+= 1
    self.timeseries_tabs.observe(lambda _: self.on_change(), names='selected_index')
    
    # Create a dropdown menu
    self.token_dropdown = widgets.Dropdown(
        options=list(self.dfs.keys()),
        value=self.params["token"],
        description='Token:',
        disabled=False,
    )


    self.token_dropdown.observe(self.on_dropdown_change, names='value')
    for slide in low_vol_spread_slides + high_vol_spread_slides+vol_bands+mavg_sliders+level_bands:
      slide.observe(self.on_slide_change, names="value")
  
  
    config_button_column = self.get_config_show_buttons()
    
    self.stress_test_scenario_column = self.stress_test_scenario()
    self.header_bottom = widgets.HBox([self.vbox_mavg_sliders, config_button_column, ])
    
    self.run_button = widgets.Button(description="Run")
    self.run_button.on_click(lambda _: self.on_change())
    self.run_button.button_style = 'primary'
    
    # self.run_button = widgets.Button(description="Run")
    # self.run_button.on_click(lambda _: self.on_change())
    # self.run_button.button_style = 'primary'
    
    self.is_running_label = widgets.Label(value="Done")


    self.header = widgets.HBox([self.token_dropdown, self.run_button, self.is_running_label])
  

  def html_page_snapshot(self, tab_number=None):
    tab_number = tab_number or self.timeseries_tabs.selected_index
    from plotly.offline import plot
    import json
    params_txt = json.dumps(self.model_params, indent=2)
    _params = pd.DataFrame([self.model_params["params"]]).T
    _params = _params.reset_index()
    _params.columns=["param", "value"]
    params_html = _params.to_html(index=False)
    
    _leverage_params = pd.DataFrame([self.model_params["leverage_params"]]).T
    _leverage_params = _leverage_params.reset_index()
    _leverage_params.columns=["leverage param", "value"]
    leverage_params_html = _leverage_params.to_html(index=False)  

    token = self.params["token"]
    contract = self.contracts[tab_number]

    fig = self.timeseries_fig[contract]

    # Convert the figure to a div element (HTML string)
    plot_div = plot(fig, output_type='div', include_plotlyjs='cdn')


    # Define the additional HTML (a paragraph and a horizontal rule)
    html_string = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>[{token.upper()}] IPOR Spread Explorer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                
            }}
            table {{
                margin-right: auto;
                margin-left: 0;
            }}
            td {{
                padding: 8px;
                text-align: left;
            }}
            button {{
            margin: 10px 0;
            padding: 10px 20px;
            font-size: 16px;
            }}
        </style>
    </head>
    <body>
        {plot_div}
        <hr>
        <button onclick="copyToClipboard()">Copy All Params</button>
        {params_html}
        <br>
        {leverage_params_html}

        <script>
        function copyToClipboard() {{
            // Create a temporary textarea element
            var textarea = document.createElement('textarea');
            // Set the value of the textarea to your dictionary string
            textarea.value = `{params_txt}`;
            // Append it to the body
            document.body.appendChild(textarea);
            // Select the textarea content
            textarea.select();
            // Execute the "Copy" command
            document.execCommand('copy');
            // Remove the textarea from the body
            document.body.removeChild(textarea);
            // Optional: alert the user that the text has been copied
            alert('Params copied to clipboard!');
        }}
    </script>
    </body>
    </html>
    """
    return html_string
    

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
        
    walktree_change_values(self.spread_section)
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
    display(self.header, self.spread_section, self.header_bottom, self.timeseries_tabs, self.stress_test_scenario_column)
    #display(tabs)
    self.on_change()