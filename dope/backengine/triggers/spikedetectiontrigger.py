import math
import numpy as np
import matplotlib.dates as mdates
import pandas as pd
from scipy.optimize import curve_fit
from dope.backengine.estimators.baseestimator import BaseEstimator
from dope.backengine.triggers.basetrigger import BaseTrigger


class SpikeDetectionTrigger(BaseTrigger):
    def __init__(self, df, lag, lagSD, est, rt_col, k=0.5, protocolList=[]):
        self.df = df
        self.lag = lag
        self.lagSD = lagSD
        self.est = est
        self.protocolList = protocolList
        self.rt_col = rt_col
        self.k = k
        self.triggerDates = None

        if not self.df.index.is_monotonic_increasing:
            #          print("WARNING: DateTimeIndex not monotonic increasing!")
            self.df = self.df.sort_index()

    def assets(self):
        protocols = [x[1] for x in list(self.df.columns) if x != "cash"]
        if self.protocolList != []:
            protocols = [v for v in protocols if v in self.protocolList and v != "cash"]
        return list(set(protocols))

    def pairs(self):
        protocols = [x[1] for x in list(self.df.columns) if x != "cash"]
        if self.protocolList != []:
            protocols = [v for v in protocols if v in self.protocolList and v != "cash"]
        N = len(protocols)
        return [
            (protocols[i], protocols[j]) for i in range(0, N) for j in range(i + 1, N)
        ]

    def rates(self):
        _df = self.df[[self.rt_col]]
        if self.lag == 0:
            mu = _df[self.rt_col]  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        mu.columns = [(self.rt_col, c + ":mu") for c in mu.columns]
        sigma.columns = [(self.rt_col, c + ":vol") for c in sigma.columns]
        t_dn.columns = [(self.rt_col, c + ":dn") for c in t_dn.columns]
        t_up.columns = [(self.rt_col, c + ":up") for c in t_up.columns]
        return pd.concat([_df, sigma, t_dn, t_up], axis=1)

    def dates(self):

        level = 0.015

        _df = self.df[[self.rt_col]]

        obs = _df[self.rt_col]

        if self.lag == 0:
            mu = _df[self.rt_col].shift(1)  # could equivalently use 'supplyRate' or borrowRate'
        else:
            mu = self.est.rolling_fit_mu(
                df=_df, lag=self.lag, rt_col=self.rt_col
            )  # could equivalently use 'supplyRate' or borrowRate'

        sigma = self.est.rolling_fit_sigma(
            df=_df, lag=self.lagSD, rt_col=self.rt_col
        )  # identify which column to use for calculating triggers

        t_dn = mu - self.k * sigma
        t_up = mu + self.k * sigma
        _filter = pd.Series(
            index=pd.Index([], dtype=int), dtype=bool
        )  # initialise to empty
        t = obs.copy()
        for asset in self.assets():
            asset = str(asset)
            t_up = t_up.copy().rename(columns={asset: asset + "_up"})
            t_dn = t_dn.copy().rename(columns={asset: asset + "_dn"})

            t[asset + "_up"] = t_up[asset + "_up"]
            t[asset + "_dn"] = t_dn[asset + "_dn"]
            t["diff_1"] = t[asset + "_up"] - t[asset]  # usually positive
            t["diff_2"] = t[asset] - t[asset + "_dn"]  # also usually positive

            t = t.fillna(
                0
            )  # if NaN, the zero will mean t[diff_1] and t[diff_2] will neither be >0 nor <0 so no trigger should eventuate

            _filter1 = t["diff_1"] < -level
            _filter1 &= t["diff_1"].shift(1) > 0

            _filter = _filter1

        self.triggerDates = list(t[_filter].index)
        self.rtcolAtTriggerDates = t[_filter]

        return self.triggerDates

    def decayParameters(self, asset):

        # Function to create segments from change points
        def create_segments(ts, xs, change_points, deviation_points=None):
            """
            Create segments from detected change points and deviation points
            """
            segments = []
            all_break_points = sorted(list(set(change_points + (deviation_points or []))))

            if not all_break_points:
                # If no break points, use the whole dataset
                segments.append((0, len(ts)))
            else:
                # First segment starts at 0
                segments.append((0, all_break_points[0]))

                # Middle segments
                for i in range(len(all_break_points) - 1):
                    segments.append((all_break_points[i], all_break_points[i + 1]))

                # Last segment ends at the end of the dataset
                segments.append((all_break_points[-1], len(ts)))

            return segments

            # Function to detect segments in the data manually based on known change point

        def detect_segments(ts, xs, change_point_index=50):
            segments = [(0, change_point_index), (change_point_index, len(ts))]
            return segments, [change_point_index]

        # Define exponential decay function with asymptote
        def exp_decay(t0, t, C, B_minus_C, k):
            """
            Exponential decay function: y = C + (B-C) * exp(-k*t)

            Parameters:
            t0, t : time points
            C : asymptotic value (plateau)
            B_minus_C : initial value minus asymptotic value
            k : decay rate
            """
            return C + B_minus_C * np.exp(-k * (t - t0) / 86400)

            # Function to fit exponential decay to a segment

        def fit_segment(ts, xs, segment):
            start_idx, end_idx = segment

            # Get segment data
            segment_ts = ts[start_idx:end_idx]
            segment_xs = xs[start_idx:end_idx]

            # Skip the first point if it's a spike
            if len(segment_ts) > 3:  # Ensure we have enough points
                segment_ts = segment_ts[1:]
                segment_xs = segment_xs[1:]

            # Check if we have enough data points after removing the spike
            if len(segment_ts) < 3:
                print(f"Not enough data points in segment {segment} after removing spike")
                return {'segment': segment, 'success': False}

            normalized_ts = [t.timestamp() for t in segment_ts]

            # Initial parameter guesses
            initial_guess = [
                min(segment_xs),  # C (plateau)
                max(segment_xs) - min(segment_xs),  # B-C (amplitude)
                0.1  # k (rate)
            ]

            # Set bounds
            bounds = (
                [0, 0, 0],  # Lower bounds
                [np.inf, np.inf, np.inf]  # Upper bounds
            )

            t0 = segment_ts[0].timestamp()

            def curried_exp_decay(t, C, B_minus_C, k):
                return exp_decay(t0, t, C, B_minus_C, k)

            try:
                # Fit the model
                popt, pcov = curve_fit(curried_exp_decay, normalized_ts, segment_xs,
                                       p0=initial_guess, bounds=bounds, maxfev=10000)

                # Extract parameters
                C_fit, B_minus_C_fit, k_fit = popt
                B_fit = C_fit + B_minus_C_fit

                # Calculate half-life
                half_life = math.log(2) / k_fit

                # Calculate R-squared
                y_pred = curried_exp_decay(np.array(normalized_ts), C_fit, B_minus_C_fit, k_fit)
                ss_total = np.sum((segment_xs - np.mean(segment_xs)) ** 2)
                ss_residual = np.sum((segment_xs - y_pred) ** 2)
                r_squared = 1 - (ss_residual / ss_total)

                area_under_curve = B_minus_C_fit / k_fit

                return {
                    'segment': segment,
                    'C': C_fit,
                    'B': B_fit,
                    'k': k_fit,
                    'half_life': half_life,
                    'r_squared': r_squared,
                    'area_under_curve': area_under_curve,
                    'success': True,
                    'normalized_ts': normalized_ts,
                    'original_ts': segment_ts,
                    'original_xs': segment_xs
                }

            except (RuntimeError, ValueError) as e:
                print(f"Fit failed for segment {segment}: {e}")
                return {
                    'segment': segment,
                    'success': False
                }

        # Function to detect model fit quality changes within a segment
        def detect_model_deviation(ts, xs, window_size=8, step=2, r_squared_threshold=0.8):
            """
            Detect where the data deviates from exponential decay model by using
            a sliding window approach and monitoring R-squared.

            Returns the index where significant deviation starts.
            """
            deviation_points = []
            r_squared_values = []

            for i in range(window_size, len(ts) - window_size, step):
                # Take increasing subsets of data
                subset_ts = ts[:i + window_size]
                subset_xs = xs[:i + window_size]

                # Skip the first point if it's likely a spike
                if abs(subset_xs[0] - np.mean(subset_xs[1:4])) > 2 * np.std(subset_xs[1:4]):
                    subset_ts = subset_ts[1:]
                    subset_xs = subset_xs[1:]

                # Normalize time
                norm_ts = [t.timestamp() for t in subset_ts]

                # Fit exponential decay model
                try:
                    t0 = subset_ts[0].timestamp()

                    def curried_exp_decay(t, C, B_minus_C, k):
                        return exp_decay(t0, t, C, B_minus_C, k)

                        # Initial parameter guesses

                    initial_guess = [
                        min(subset_xs),  # C (plateau)
                        max(subset_xs) - min(subset_xs),  # B-C (amplitude)
                        0.1  # k (rate)
                    ]

                    # Fit the model
                    popt, _ = curve_fit(curried_exp_decay, norm_ts, subset_xs,
                                        p0=initial_guess, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]),
                                        maxfev=5000)

                    # Calculate R-squared
                    y_pred = curried_exp_decay(np.array(norm_ts), *popt)
                    ss_total = np.sum((subset_xs - np.mean(subset_xs)) ** 2)
                    ss_residual = np.sum((subset_xs - y_pred) ** 2)
                    r_squared = 1 - (ss_residual / ss_total)

                    r_squared_values.append((i, r_squared))

                    # If R-squared drops below threshold, mark as deviation point
                    if r_squared < r_squared_threshold and i > 15:  # Ensure we're not at the very beginning
                        deviation_points.append(i)  #  MAU TODO - check deviation points for the array length of the look-forward windows
                                                    #  - note this is the number of array elements, NOT a datetime interval
                        if len(deviation_points) >= 3:  # Require multiple consistent deviation signals
                            break

                except (RuntimeError, ValueError):
                    # If fit fails, this could indicate a deviation point
                    deviation_points.append(i)   # MAU TODO - check deviation points for the array length of the look-forward windows
                                                 # - note this is the number of array elements, NOT a datetime interval

            # Return first consistent deviation point if found
            if deviation_points and len(deviation_points) >= 3:
                return deviation_points[0], r_squared_values
            else:
                return None, r_squared_values

        df = self.rtcolAtTriggerDates[[asset]].reset_index()
        df['B'] = df[asset]
        df = df.set_index('datetime')
        del df[asset]
        df['C'] = np.nan
        df['k'] = np.nan

        df2 = pd.DataFrame(df, columns=['B', 'C', 'k'])

        for start in self.triggerDates:
            _df = self.df[start:][[self.rt_col]]
            obs = _df[self.rt_col][asset]

            ts = list(_df[self.rt_col].index)
            xs = list(obs)

            # First segment to check for deviation from exponential decay
            deviation_point, r_squared_values = detect_model_deviation(
                ts,
                xs,
                window_size=8,
                step=1,
                r_squared_threshold=0.95
            )

            deviation_points = []
            if deviation_point is not None:
                deviation_points = [deviation_point]
            else:
                pass

            result = fit_segment(ts, xs, (0, deviation_point))

            df2.loc[df2.index == start, 'C'] = result['C']
            df2.loc[df2.index == start, 'k'] = result['k']

        return df2

