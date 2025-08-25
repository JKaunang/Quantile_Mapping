import numpy as np
import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Toplevel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import ks_2samp, pearsonr
from datetime import datetime
import os
import math

class IDWRainfallAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("IDW Rainfall Analysis")
        self.root.geometry("800x800")
        self.style = ttk.Style(theme="darkly")
        self.loc1_df = None
        self.loc2_df = None
        self.loc3_ground_df = None
        self.metrics = {}
        self.plots = {}
        self.output_dir_var = ttk.StringVar(value="")
        self.loc1_coords = ttk.StringVar(value="0,0")
        self.loc2_coords = ttk.StringVar(value="0,0")
        self.loc3_coords = ttk.StringVar(value="0,0")
        self.agg_level_var = ttk.StringVar(value="Yearly")
        self.agg_func_var = ttk.StringVar(value="Max")
        self.idw_power_var = ttk.StringVar(value="2.0")
        self.threshold_var = ttk.StringVar(value="50th")
        self.log_file_path = None
        self.merged_data = None
        self.storm_threshold_mm = None
        self.setup_gui()

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Input")
        self.setup_data_tab()
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Plots")
        self.setup_plots_tab()

    def setup_data_tab(self):
        container = ttk.Frame(self.data_frame)
        container.pack(pady=10, padx=10, fill="both", expand=True)
        ttk.Label(container, text="Location 1 QM Output File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.loc1_entry = ttk.Entry(container, width=50)
        self.loc1_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=lambda: self.load_data(1)).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(container, text="Location 2 QM Output File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.loc2_entry = ttk.Entry(container, width=50)
        self.loc2_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=lambda: self.load_data(2)).grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(container, text="Location 3 Ground Data File:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.loc3_entry = ttk.Entry(container, width=50)
        self.loc3_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=lambda: self.load_data(3)).grid(row=2, column=2, padx=5, pady=5)
        ttk.Label(container, text="Output Directory:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.output_dir_entry = ttk.Entry(container, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=self.browse_output_dir).grid(row=3, column=2, padx=5, pady=5)
        ttk.Label(container, text="Location 1 Coordinates (lat,lon):").grid(row=4, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(container, textvariable=self.loc1_coords, width=20).grid(row=4, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="Location 2 Coordinates (lat,lon):").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(container, textvariable=self.loc2_coords, width=20).grid(row=5, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="Location 3 Coordinates (lat,lon):").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(container, textvariable=self.loc3_coords, width=20).grid(row=6, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="Aggregation Level:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.agg_level_combo = ttk.Combobox(
            container, textvariable=self.agg_level_var,
            values=["Daily", "Monthly", "Yearly"], state="readonly"
        )
        self.agg_level_combo.grid(row=7, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="Aggregation Function:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.agg_func_combo = ttk.Combobox(
            container, textvariable=self.agg_func_var,
            values=["Mean", "Max", "Total"], state="readonly"
        )
        self.agg_func_combo.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="IDW Power:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.idw_power_entry = ttk.Entry(container, textvariable=self.idw_power_var, width=10)
        self.idw_power_entry.grid(row=9, column=1, padx=5, pady=5, sticky="w")
        ttk.Label(container, text="Event Threshold:").grid(row=10, column=0, padx=5, pady=5, sticky="w")
        self.threshold_combo = ttk.Combobox(
            container, textvariable=self.threshold_var,
            values=["25th", "50th", "75th", "90th", "95th"], state="readonly"
        )
        self.threshold_combo.grid(row=10, column=1, padx=5, pady=5, sticky="w")
        ttk.Button(container, text="Run Analysis", command=self.run_single_analysis, style="primary.TButton").grid(
            row=11, column=0, pady=10
        )
        ttk.Button(container, text="Optimize Power", command=self.optimize_power, style="primary.TButton").grid(
            row=11, column=1, pady=10, sticky="w"
        )
        ttk.Button(container, text="Batch Run", command=self.batch_run, style="secondary.TButton").grid(
            row=11, column=2, pady=10
        )
        ttk.Button(container, text="Reset", command=self.reset).grid(row=12, column=1, pady=10)
        log_frame = ttk.Frame(container)
        log_frame.grid(row=13, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.log_text = ttk.Text(log_frame, height=10, width=80, wrap="word")
        self.log_text.pack(side="top", fill="both", expand=True)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        self.log(f"Application started at {current_time}.")

    def setup_results_tab(self):
        self.tree = ttk.Treeview(self.results_frame, columns=("Metric", "IDW"), show="headings")
        self.tree.heading("Metric", text="Metric")
        self.tree.heading("IDW", text="IDW")
        self.tree.pack(pady=10, padx=10, fill="both", expand=True)
        ttk.Button(self.results_frame, text="Save Metrics", command=self.save_metrics).pack(pady=5)

    def setup_plots_tab(self):
        self.plot_var = ttk.StringVar(value="Time Series")
        self.plot_combobox = ttk.Combobox(
            self.plots_frame, textvariable=self.plot_var, 
            values=["Histogram", "CDF", "Time Series", "Scatter"])
        self.plot_combobox.pack(pady=5)
        self.plot_combobox.bind("<<ComboboxSelected>>", self.display_plot)
        self.canvas_frame = ttk.Frame(self.plots_frame)
        self.canvas_frame.pack(pady=10, fill="both", expand=True)
        ttk.Button(self.plots_frame, text="Save Plot", command=self.save_plot).pack(pady=5)

    def browse_output_dir(self):
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
            self.log(f"Output directory set to: {directory}")

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                self.log(f"Error writing to log file: {e}")

    def select_columns(self, df, file_path, loc_number):
        columns = list(df.columns)
        if not columns:
            raise ValueError("No columns found in file")
        time_col = [None]
        precip_col = [None]
        top = Toplevel(self.root)
        top.title(f"Select Columns for Location {loc_number}")
        ttk.Label(top, text=f"Select columns for {file_path}:").pack(pady=5)
        ttk.Label(top, text="Date Column:").pack(pady=5)
        time_var = ttk.StringVar()
        time_combo = ttk.Combobox(top, textvariable=time_var, values=columns, state="readonly")
        time_combo.pack(pady=5)
        time_combo.set(columns[0] if columns else "")
        ttk.Label(top, text="Precipitation Column:").pack(pady=5)
        precip_var = ttk.StringVar()
        if loc_number == 3:
            values = [col for col in columns if col != time_var.get()]
        else:
            values = ["Satellite", "Satellite_Linear", "Satellite_Rank", "Satellite_Spline"]
            values = [v for v in values if v in columns]
        precip_combo = ttk.Combobox(top, textvariable=precip_var, values=values, state="readonly")
        precip_combo.pack(pady=5)
        precip_combo.set(values[0] if values else "")
        
        def confirm():
            time_col[0] = time_var.get()
            precip_col[0] = precip_var.get()
            if not time_col[0] or not precip_col[0]:
                messagebox.showerror("Error", "Please select both date and precipitation columns")
                top.destroy()
                return
            if time_col[0] == precip_col[0]:
                messagebox.showerror("Error", "Date and precipitation columns must be different")
                top.destroy()
                return
            top.destroy()

        ttk.Button(top, text="Confirm", command=confirm).pack(side="left", padx=10, pady=10)
        ttk.Button(top, text="Cancel", command=lambda: top.destroy()).pack(side="right", padx=10, pady=10)
        top.grab_set()
        self.root.wait_window(top)
        
        if not time_col[0] or not precip_col[0]:
            raise ValueError("Column selection cancelled or invalid")
        return time_col[0], precip_col[0]

    def load_data(self, loc_number):
        file_path = filedialog.askopenfilename(filetypes=[("Excel/CSV files", "*.xlsx *.csv")])
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_csv(file_path)
                time_col, precip_col = self.select_columns(df, file_path, loc_number)
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if df[time_col].isna().any():
                    raise ValueError("Invalid or missing dates in date column")
                if not pd.to_numeric(df[precip_col], errors='coerce').notna().any():
                    raise ValueError("Precipitation column contains no valid numeric values")
                df = df[[time_col, precip_col]].rename(columns={time_col: 'Date', precip_col: 'Value'})
                if loc_number == 1:
                    self.loc1_df = df
                    self.loc1_entry.delete(0, "end")
                    self.loc1_entry.insert(0, file_path)
                    self.log(f"Loaded Location 1 QM data: {file_path} (Date: {time_col}, Precipitation: {precip_col})")
                elif loc_number == 2:
                    self.loc2_df = df
                    self.loc2_entry.delete(0, "end")
                    self.loc2_entry.insert(0, file_path)
                    self.log(f"Loaded Location 2 QM data: {file_path} (Date: {time_col}, Precipitation: {precip_col})")
                else:
                    self.loc3_ground_df = df
                    self.loc3_entry.delete(0, "end")
                    self.loc3_entry.insert(0, file_path)
                    self.log(f"Loaded Location 3 ground data: {file_path} (Date: {time_col}, Precipitation: {precip_col})")
                self.aggregate_data(loc_number)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load data for Location {loc_number}: {e}")
                self.log(f"Error loading data for Location {loc_number}: {e}")

    def aggregate_data(self, loc_number):
        df = self.loc1_df if loc_number == 1 else self.loc2_df if loc_number == 2 else self.loc3_ground_df
        if df is None:
            return
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        daily = df[['Date', 'Value']].sort_values('Date').dropna(subset=['Value'])
        monthly = df.groupby(['Year', 'Month'], as_index=False)['Value'].agg(['mean', 'max', 'sum'])
        monthly.columns = ['Year', 'Month', 'Mean', 'Max', 'Total']
        monthly['Date'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str) + '-01')
        monthly = monthly.dropna(subset=['Mean', 'Max', 'Total'])
        yearly = df.groupby('Year', as_index=False)['Value'].agg(['mean', 'max', 'sum'])
        yearly.columns = ['Year', 'Mean', 'Max', 'Total']
        yearly['Date'] = pd.to_datetime(yearly['Year'].astype(str) + '-01-01')
        yearly = yearly.dropna(subset=['Mean', 'Max', 'Total'])
        if loc_number == 1:
            self.loc1_daily, self.loc1_monthly, self.loc1_yearly = daily, monthly, yearly
            self.log(f"Location 1 aggregation: {len(daily)} daily, {len(monthly)} monthly, {len(yearly)} yearly rows")
        elif loc_number == 2:
            self.loc2_daily, self.loc2_monthly, self.loc2_yearly = daily, monthly, yearly
            self.log(f"Location 2 aggregation: {len(daily)} daily, {len(monthly)} monthly, {len(yearly)} yearly rows")
        else:
            self.loc3_daily, self.loc3_monthly, self.loc3_yearly = daily, monthly, yearly
            self.log(f"Location 3 aggregation: {len(daily)} daily, {len(monthly)} monthly, {len(yearly)} yearly rows")

    def get_selected_agg_data(self, loc_number):
        level = self.agg_level_var.get()
        func = self.agg_func_var.get()
        col = {"Mean": "Mean", "Max": "Max", "Total": "Total"}.get(func, "Mean")
        if loc_number == 1:
            data = (self.loc1_daily if level == "Daily" else 
                    self.loc1_monthly[["Date", col]].rename(columns={col: "Value"}) if level == "Monthly" else 
                    self.loc1_yearly[["Date", col]].rename(columns={col: "Value"}))
        elif loc_number == 2:
            data = (self.loc2_daily if level == "Daily" else 
                    self.loc2_monthly[["Date", col]].rename(columns={col: "Value"}) if level == "Monthly" else 
                    self.loc2_yearly[["Date", col]].rename(columns={col: "Value"}))
        else:
            data = (self.loc3_daily if level == "Daily" else 
                    self.loc3_monthly[["Date", col]].rename(columns={col: "Value"}) if level == "Monthly" else 
                    self.loc3_yearly[["Date", col]].rename(columns={col: "Value"}))
        data = data.dropna(subset=["Value"])
        self.log(f"Selected Location {loc_number} {level} {func} data: {len(data)} rows, {data['Value'].isna().sum()} NaNs")
        return data

    def reset(self):
        self.loc1_entry.delete(0, "end")
        self.loc2_entry.delete(0, "end")
        self.loc3_entry.delete(0, "end")
        self.output_dir_var.set("")
        self.loc1_coords.set("0,0")
        self.loc2_coords.set("0,0")
        self.loc3_coords.set("0,0")
        self.agg_level_var.set("Yearly")
        self.agg_func_var.set("Max")
        self.idw_power_var.set("2.0")
        self.threshold_var.set("50th")
        self.loc1_df = self.loc2_df = self.loc3_ground_df = None
        self.metrics.clear()
        self.plots.clear()
        self.merged_data = None
        self.storm_threshold_mm = None
        self.log_text.delete(1.0, "end")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
        self.log(f"Application reset at {current_time}.")

    def parse_coordinates(self, coord_str):
        try:
            lat, lon = map(float, coord_str.split(','))
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude {lat} out of range [-90, 90]")
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude {lon} out of range [-180, 180]")
            return lat, lon
        except Exception as e:
            raise ValueError(f"Invalid coordinates format: {coord_str}. Expected 'lat,lon'. {str(e)}")

    def idw_interpolation(self, merged_train, loc3_test, loc1_coords, loc2_coords, loc3_coords, power):
        lat1, lon1 = loc1_coords
        lat2, lon2 = loc2_coords
        lat3, lon3 = loc3_coords
        self.log(f"Raw coordinates: Loc1 ({lat1:.6f}, {lon1:.6f}), Loc2 ({lat2:.6f}, {lon2:.6f}), Loc3 ({lat3:.6f}, {lon3:.6f})")
        
        lat1, lon1, lat2, lon2, lat3, lon3 = map(math.radians, [lat1, lon1, lat2, lon2, lat3, lon3])
        dlat1 = lat3 - lat1
        dlon1 = lon3 - lon1
        dlat2 = lat3 - lat2
        dlon2 = lon3 - lon2
        self.log(f"Raw coordinate differences (radians): Loc1-Loc3 ({dlat1:.6f}, {dlon1:.6f}), Loc2-Loc3 ({dlat2:.6f}, {dlon2:.6f})")
        
        R = 6371.0
        a1 = math.sin(dlat1 / 2)**2 + math.cos(lat1) * math.cos(lat3) * math.sin(dlon1 / 2)**2
        c1 = 2 * math.atan2(math.sqrt(a1), math.sqrt(1 - a1))
        d1 = R * c1
        a2 = math.sin(dlat2 / 2)**2 + math.cos(lat2) * math.cos(lat3) * math.sin(dlon2 / 2)**2
        c2 = 2 * math.atan2(math.sqrt(a2), math.sqrt(1 - a2))
        d2 = R * c2
        self.log(f"Haversine components (Loc1-Loc3): a1={a1:.6e}, c1={c1:.6f}, d1={d1:.2f} km")
        self.log(f"Haversine components (Loc2-Loc3): a2={a2:.6e}, c2={c2:.6f}, d2={d2:.2f} km")
        
        if d1 == 0 or d2 == 0:
            self.log("Warning: One or more locations have identical coordinates")
            return None

        common_dates = loc3_test['Date']
        merged = merged_train[merged_train['Date'].isin(common_dates)]
        if merged.empty:
            self.log("No common dates found for interpolation")
            return None

        interpolated = []
        for date in loc3_test['Date']:
            data = merged[merged['Date'] == date]
            if len(data) == 0:
                interpolated.append(np.nan)
                continue
            v1, v2 = data['Value_loc1'].iloc[0], data['Value_loc2'].iloc[0]
            w1 = 1 / (d1 ** power) if d1 != 0 else np.inf
            w2 = 1 / (d2 ** power) if d2 != 0 else np.inf
            w_sum = w1 + w2
            if w_sum == 0 or np.isinf(w_sum):
                interpolated.append(np.nan)
            else:
                interpolated.append((v1 * w1 + v2 * w2) / w_sum)
        
        result = pd.DataFrame({'Date': loc3_test['Date'], 'IDW': interpolated})
        result = result.dropna()
        self.log(f"IDW interpolated {len(result)} values, mean={result['IDW'].mean():.2f} mm")
        return result

    def calculate_metrics(self, observed, predicted):
        observed = observed.dropna()
        predicted = predicted.dropna()
        common_idx = observed.index.intersection(predicted.index)
        if len(common_idx) == 0:
            self.log("No common dates for metrics calculation")
            return {}
        observed = observed.loc[common_idx]
        predicted = predicted.loc[common_idx]
        
        rmse = np.sqrt(mean_squared_error(observed, predicted))
        mae = mean_absolute_error(observed, predicted)
        bias = np.mean(predicted - observed)
        ks_stat, _ = ks_2samp(observed, predicted)
        
        threshold_map = {"25th": 25, "50th": 50, "75th": 75, "90th": 90, "95th": 95}
        percentile = threshold_map.get(self.threshold_var.get(), 50)
        storm_threshold = np.percentile(observed, percentile) if len(observed) > 0 else 20.0
        self.storm_threshold_mm = storm_threshold
        self.log(f"Event detection threshold ({self.threshold_var.get()} percentile): {storm_threshold:.2f} mm")
        
        rain_events_obs = observed > storm_threshold
        rain_events_pred = predicted > storm_threshold
        hits = np.sum(rain_events_obs & rain_events_pred)
        misses = np.sum(rain_events_obs & ~rain_events_pred)
        false_alarms = np.sum(~rain_events_obs & rain_events_pred)
        pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
        
        ss_tot = np.sum((observed - np.mean(observed))**2)
        ss_res = np.sum((observed - predicted)**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        if len(observed) >= 2:
            pearson_corr, _ = pearsonr(observed, predicted)
        else:
            pearson_corr = np.nan
        
        return {
            'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'POD': pod, 'FAR': far, 'KS': ks_stat,
            'R2': r2, 'Pearson': pearson_corr
        }

    def generate_plots(self, data, timestamp):
        os.makedirs(self.output_dir_var.get(), exist_ok=True)
        self.plots.clear()
        fig_hist = plt.figure(figsize=(12, 6))
        bins = int(np.sqrt(len(data['IDW'])))
        bin_edges = np.histogram_bin_edges(data['IDW'].dropna(), bins=bins)
        plt.hist(data['Ground'].dropna(), bins=bin_edges, alpha=0.5, label='Observed')
        plt.hist(data['IDW'].dropna(), bins=bin_edges, alpha=0.5, label='IDW Interpolated')
        plt.legend()
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Frequency')
        plt.title('Histogram of Observed vs. IDW Interpolated Rainfall')
        self.plots['Histogram'] = fig_hist
        plt.savefig(os.path.join(self.output_dir_var.get(), f'histogram_idw_{timestamp}.png'))
        plt.close(fig_hist)
        fig_cdf = plt.figure(figsize=(12, 6))
        x_g, y_g = np.sort(data['Ground'].dropna()), np.arange(1, len(data['Ground'].dropna()) + 1) / len(data['Ground'].dropna())
        x_i, y_i = np.sort(data['IDW'].dropna()), np.arange(1, len(data['IDW'].dropna()) + 1) / len(data['IDW'].dropna())
        plt.plot(x_g, y_g, label='Observed')
        plt.plot(x_i, y_i, label='IDW Interpolated')
        plt.legend()
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical CDFs')
        plt.grid(True)
        self.plots['CDF'] = fig_cdf
        plt.savefig(os.path.join(self.output_dir_var.get(), f'cdf_idw_{timestamp}.png'))
        plt.close(fig_cdf)
        fig_ts = plt.figure(figsize=(12, 6))
        plt.plot(data['Date'], data['Ground'], label='Observed', color='blue', alpha=0.5)
        plt.plot(data['Date'], data['IDW'], label='IDW Interpolated', color='orange', alpha=0.5)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.title('Time Series Comparison')
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.plots['Time Series'] = fig_ts
        plt.savefig(os.path.join(self.output_dir_var.get(), f'timeseries_idw_{timestamp}.png'))
        plt.close(fig_ts)
        fig_scatter = plt.figure(figsize=(12, 6))
        plt.scatter(data['Ground'], data['IDW'], alpha=0.5, color='blue')
        min_val = min(data['Ground'].min(), data['IDW'].min())
        max_val = max(data['Ground'].max(), data['IDW'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
        plt.text(0.05, 0.95, f'R²={self.metrics.get("R2"):.3f}\nPearson={self.metrics.get("Pearson"):.3f}',
                 transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        plt.legend()
        plt.xlabel('Observed Rainfall (mm)')
        plt.ylabel('IDW Interpolated Rainfall (mm)')
        plt.title('Scatter Plot: Observed vs. IDW Interpolated Rainfall')
        plt.grid(True)
        self.plots['Scatter'] = fig_scatter
        plt.savefig(os.path.join(self.output_dir_var.get(), f'scatter_idw_{timestamp}.png'))
        plt.close(fig_scatter)
        self.log(f"Generated plots: {list(self.plots.keys())}")

    def display_plot(self, event=None):
        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            plot_name = self.plot_var.get()
            if plot_name in self.plots:
                canvas = FigureCanvasTkAgg(self.plots[plot_name], master=self.canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.log(f"Displayed {plot_name} plot")
            else:
                messagebox.showerror("Error", f"Plot {plot_name} not available. Run analysis first.")
        except Exception as e:
            self.log(f"Error displaying {plot_name}: {e}")
            messagebox.showerror("Error", f"Failed to display {plot_name}: {e}")

    def save_metrics(self):
        file_path = filedialog.asksaveasfilename(
            initialdir=self.output_dir_var.get(),
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if file_path:
            metrics_data = {
                'Metric': list(self.metrics.keys()),
                'IDW': [self.metrics.get(m, np.nan) for m in self.metrics.keys()]
            }
            pd.DataFrame(metrics_data).to_csv(file_path, index=False)
            self.log(f"Metrics saved to: {file_path}")

    def save_plot(self):
        try:
            plot_name = self.plot_var.get()
            if plot_name in self.plots:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = filedialog.asksaveasfilename(
                    initialdir=self.output_dir_var.get(),
                    initialfile=f"{plot_name.lower().replace(' ', '_')}_{timestamp}.png",
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png")]
                )
                if file_path:
                    self.plots[plot_name].savefig(file_path)
                    self.log(f"Plot {plot_name} saved to: {file_path}")
            else:
                messagebox.showerror("Error", f"Plot {plot_name} not available.")
        except Exception as e:
            self.log(f"Error saving plot {plot_name}: {e}")
            messagebox.showerror("Error", f"Failed to save {plot_name}: {e}")

    def plot_metric_trends(self, batch_results, timestamp):
        powers = [r['Power'] for r in batch_results]
        metrics = ['RMSE', 'MAE', 'Bias', 'R2', 'Pearson', 'POD', 'FAR', 'KS']
        fig = plt.figure(figsize=(12, 6))
        for metric in metrics:
            values = [r[metric] for r in batch_results]
            plt.plot(powers, values, label=metric, marker='o')
        plt.xlabel('IDW Power')
        plt.ylabel('Metric Value')
        plt.title('Metric Trends vs. IDW Power')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir_var.get(), f'metric_trends_{timestamp}.png'))
        plt.close(fig)
        self.log(f"Metric trends plot saved to: metric_trends_{timestamp}.png")

    def optimize_power(self):
        try:
            if not all([df is not None and not df.empty for df in [self.loc1_df, self.loc2_df, self.loc3_ground_df]]):
                raise ValueError("All data files must be loaded and non-empty")
            
            loc1_data = self.get_selected_agg_data(1)
            loc2_data = self.get_selected_agg_data(2)
            loc3_data = self.get_selected_agg_data(3)
            merged_train = pd.merge(loc1_data, loc2_data, on='Date', how='inner', suffixes=('_loc1', '_loc2'))
            
            loc1_coords = self.parse_coordinates(self.loc1_coords.get())
            loc2_coords = self.parse_coordinates(self.loc2_coords.get())
            loc3_coords = self.parse_coordinates(self.loc3_coords.get())
            
            best_power, best_rmse = None, float('inf')
            powers = np.arange(1.0, 10.1, 0.5)
            for power in powers:
                result = self.idw_interpolation(
                    merged_train, loc3_data, 
                    loc1_coords, loc2_coords, loc3_coords, power
                )
                if result is None:
                    self.log(f"IDW failed for power={power:.2f}")
                    continue
                merged = pd.merge(result, loc3_data, on='Date', 
                                 suffixes=('_idw', '_ground')).rename(columns={'Value': 'Ground', 'IDW': 'IDW'})
                metrics = self.calculate_metrics(merged['Ground'], merged['IDW'])
                if metrics.get('RMSE', float('inf')) < best_rmse:
                    best_rmse, best_power = metrics['RMSE'], power
            if best_power is None:
                best_power = 2.0
            self.idw_power_var.set(f"{best_power:.2f}")
            self.log(f"Optimal power: {best_power:.2f} with RMSE: {best_rmse:.3f}")
            messagebox.showinfo("Optimization Complete", f"Optimal power: {best_power:.2f} with RMSE: {best_rmse:.3f}")
        except Exception as e:
            self.log(f"Error in power optimization: {str(e)}")
            messagebox.showerror("Error", f"Power optimization failed: {str(e)}")

    def batch_run(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            batch_log_file = os.path.join(self.output_dir_var.get(), f"batch_log_{timestamp}.txt")
            self.log(f"Starting batch run. Logs saved to: {batch_log_file}")
            
            loc1_data = self.get_selected_agg_data(1)
            loc2_data = self.get_selected_agg_data(2)
            loc3_data = self.get_selected_agg_data(3)
            merged_train = pd.merge(loc1_data, loc2_data, on='Date', how='inner', suffixes=('_loc1', '_loc2'))
            
            loc1_coords = self.parse_coordinates(self.loc1_coords.get())
            loc2_coords = self.parse_coordinates(self.loc2_coords.get())
            loc3_coords = self.parse_coordinates(self.loc3_coords.get())
            
            powers = np.arange(1.0, 10.1, 0.5)
            batch_results = []
            batch_data = {}
            
            for power in powers:
                self.log(f"Running IDW for power={power:.2f}")
                result = self.idw_interpolation(
                    merged_train, loc3_data, 
                    loc1_coords, loc2_coords, loc3_coords, power
                )
                if result is None:
                    self.log(f"IDW failed for power={power:.2f}")
                    continue
                merged = pd.merge(result, loc3_data, on='Date', 
                                 suffixes=('_idw', '_ground')).rename(columns={'Value': 'Ground', 'IDW': 'IDW'})
                metrics = self.calculate_metrics(merged['Ground'], merged['IDW'])
                batch_results.append({
                    'Power': power,
                    'RMSE': metrics.get('RMSE'), 'MAE': metrics.get('MAE'), 'Bias': metrics.get('Bias'),
                    'POD': metrics.get('POD'), 'FAR': metrics.get('FAR'), 'KS': metrics.get('KS'),
                    'R2': metrics.get('R2'), 'Pearson': metrics.get('Pearson')
                })
                batch_data[f"Power_{power:.2f}"] = merged
                
                self.log(f"Power={power:.2f} Metrics: {metrics}")
                self.generate_plots(merged, f"{timestamp}_power{power:.2f}")
            
            self.plot_metric_trends(batch_results, timestamp)
            
            output_file = os.path.join(self.output_dir_var.get(), f"batch_results_{timestamp}.xlsx")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                pd.DataFrame(batch_results).to_excel(writer, sheet_name='BatchMetrics', index=False)
                for power, data in batch_data.items():
                    data.to_excel(writer, sheet_name=power.replace('.', '_'), index=False)
                pd.DataFrame({
                    'Parameter': ['Aggregation Level', 'Aggregation Function', 
                                  'Loc1 Coords', 'Loc2 Coords', 'Loc3 Coords', 'Threshold'],
                    'Value': [self.agg_level_var.get(), self.agg_func_var.get(), 
                              self.loc1_coords.get(), self.loc2_coords.get(), 
                              self.loc3_coords.get(), self.threshold_var.get()]
                }).to_excel(writer, sheet_name='Parameters', index=False)
            
            self.log(f"Batch run complete. Results saved to: {output_file}")
            messagebox.showinfo("Batch Run Complete", f"Results saved to: {output_file}")
        except Exception as e:
            self.log(f"Error in batch run: {str(e)}")
            messagebox.showerror("Error", f"Batch run failed: {str(e)}")

    def run_single_analysis(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            self.log_file_path = os.path.join(self.output_dir_var.get(), f"idw_log_{timestamp}.txt")
            self.log(f"Starting IDW analysis. Logs saved to: {self.log_file_path}")

            if not all([df is not None and not df.empty for df in [self.loc1_df, self.loc2_df, self.loc3_ground_df]]):
                raise ValueError("All data files must be loaded and non-empty")

            loc1_data = self.get_selected_agg_data(1)
            loc2_data = self.get_selected_agg_data(2)
            loc3_data = self.get_selected_agg_data(3)
            merged_train = pd.merge(loc1_data, loc2_data, on='Date', how='inner', suffixes=('_loc1', '_loc2'))
            
            loc1_coords = self.parse_coordinates(self.loc1_coords.get())
            loc2_coords = self.parse_coordinates(self.loc2_coords.get())
            loc3_coords = self.parse_coordinates(self.loc3_coords.get())
            self.log(f"Coordinates - Loc1: {loc1_coords}, Loc2: {loc2_coords}, Loc3: {loc3_coords}")

            try:
                idw_power = float(self.idw_power_var.get())
                if idw_power <= 0:
                    raise ValueError("IDW power must be positive")
                self.log(f"IDW Power set to: {idw_power:.2f}")
            except ValueError as e:
                raise ValueError(f"Invalid IDW power value: {str(e)}. Please provide a positive number.")

            interpolated = self.idw_interpolation(
                merged_train, loc3_data, 
                loc1_coords, loc2_coords, loc3_coords, idw_power
            )
            if interpolated is None:
                raise ValueError("IDW interpolation failed")

            merged = pd.merge(interpolated, loc3_data, on='Date', how='inner', 
                             suffixes=('_idw', '_ground')).rename(columns={'Value': 'Ground', 'IDW': 'IDW'})
            self.merged_data = merged
            self.log(f"Merged data for evaluation: {len(merged)} rows")

            self.metrics = self.calculate_metrics(merged['Ground'], merged['IDW'])
            self.log(f"Metrics: {self.metrics}")

            self.generate_plots(merged, timestamp)

            self.tree.delete(*self.tree.get_children())
            for metric, value in self.metrics.items():
                self.tree.insert("", "end", values=[metric, f"{value:.3f}" if not pd.isna(value) else "NaN"])

            output_file = os.path.join(self.output_dir_var.get(), f"idw_results_{timestamp}.xlsx")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                merged.to_excel(writer, sheet_name='Data', index=False)
                pd.DataFrame({'Metric': list(self.metrics.keys()), 'Value': list(self.metrics.values())}).to_excel(
                    writer, sheet_name='Metrics', index=False
                )
                pd.DataFrame({
                    'Parameter': ['Aggregation Level', 'Aggregation Function', 
                                  'Loc1 Coords', 'Loc2 Coords', 'Loc3 Coords', 
                                  'IDW Power', 'Threshold', 'Threshold (mm)'],
                    'Value': [self.agg_level_var.get(), self.agg_func_var.get(), 
                              self.loc1_coords.get(), self.loc2_coords.get(), 
                              self.loc3_coords.get(), f"{idw_power:.2f}", 
                              self.threshold_var.get(), f"{self.storm_threshold_mm:.2f}" if self.storm_threshold_mm else "N/A"]
                }).to_excel(writer, sheet_name='Parameters', index=False)
            self.log(f"Analysis complete. Results saved to: {output_file}")

            self.display_plot(None)
        except Exception as e:
            self.log(f"Error in IDW analysis: {str(e)}")
            messagebox.showerror("Error", f"IDW analysis failed: {str(e)}")

if __name__ == "__main__":
    root = ttk.Window()
    app = IDWRainfallAnalysisGUI(root)
    root.mainloop()