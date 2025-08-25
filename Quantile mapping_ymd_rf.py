'''
Quantile Mapping Analysis for Bias Correction of Satellite Rainfall Data based on Ground Measurements
By: James Albert Kaunang
Last Updated: June 25, 2025
Changelog:
- Added option to analyze data monthly or yearly by aggregating daily data
- Added option to select between original and simplified cubic spline quantile mapping
- Implemented simplified spline using fixed percentile-based knots and non-zero data
- Retained original spline with adaptive knot selection
- Updated GUI and analysis pipeline to support spline method selection
- Added option to select aggregation level (daily, monthly, yearly) and function (mean, max)
'''

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.stats import ks_2samp, rankdata, gamma, iqr
import warnings
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Toplevel
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import openpyxl
from datetime import datetime
from sklearn.linear_model import LogisticRegression
import sys
from io import StringIO
import uuid
import os
import re
from sklearn.preprocessing import StandardScaler

# Constants
DEFAULT_STORM_THRESHOLD = 20.0
MIN_RAINFALL_THRESHOLD = 1  # Values below this are set to 0
QUARTILE_OPTIONS = {
    "Quartiles": [25, 50, 75],
    "Quintiles": [10, 25, 50, 75, 90],
    "Deciles": [10, 20, 30, 40, 50, 60, 70, 80, 90],
    "Percentiles": list(np.arange(1, 101)),
    "Custom": None
}
DRY_DAY_DISPARITY_THRESHOLD = 0.2  # Warn if dry day fraction difference > 20%
np.random.seed(42)

class RainfallAnalysisGUI:
    def __init__(self, root):
        self.diagnostic_results = {}  # Store method diagnostics
        self.root = root
        self.root.title("Rainfall Quantile mapping Analysis")
        self.root.geometry("700x700")
        self.style = ttk.Style(theme="darkly")
        self.sat_df = None
        self.ground_df = None
        self.data = None
        self.metrics = {}
        self.plots = {}
        self.zero_rain_var = ttk.BooleanVar(value=True)
        self.outlier_capping_var = ttk.BooleanVar(value=True)
        self.start_date_var = ttk.StringVar()
        self.end_date_var = ttk.StringVar()
        self.spline_method_var = ttk.StringVar(value="Original")
        self.bias_correction_scope_var = ttk.StringVar(value="Common Dates")    
        self.output_dir_var = ttk.StringVar(value="") 
        self.log_file_path = None  # Initialize log file path
        self.setup_gui()

    def browse_output_dir(self):
        """Browse for output directory."""
        directory = filedialog.askdirectory(initialdir=self.output_dir_var.get())
        if directory:
            self.output_dir_var.set(directory)
            self.log(f"Output directory set to: {directory}")

    def store_diagnostic(self, method_name, diagnostics):
        self.diagnostic_results[method_name] = diagnostics

    def get_diagnostic(self, method_name, key, default=None):
        return self.diagnostic_results.get(method_name, {}).get(key, default)

    def setup_gui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # Data Input tab with ScrolledFrame
        self.data_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.data_frame, text="Data Input")
        self.setup_data_tab()

        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()

        # Plots tab
        self.plots_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plots_frame, text="Plots")
        self.setup_plots_tab()

    def setup_data_tab(self):
        # Create a Canvas for scrolling
        canvas = ttk.Canvas(self.data_frame)
        canvas.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Add a vertical Scrollbar
        scrollbar = ttk.Scrollbar(self.data_frame, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Create a frame inside the Canvas to hold widgets
        container = ttk.Frame(canvas)
        canvas_frame = canvas.create_window((0, 0), window=container, anchor="nw")

        # Update scroll region when container size changes
        def configure_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        container.bind("<Configure>", configure_scroll_region)

        # Handle mouse wheel scrolling
        def on_mouse_wheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")

        canvas.bind_all("<MouseWheel>", on_mouse_wheel)

        # Satellite Data File
        ttk.Label(container, text="Satellite Data File:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.sat_entry = ttk.Entry(container, width=50)
        self.sat_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=self.load_sat_data).grid(row=0, column=2, padx=5, pady=5)

        # Ground Data File
        ttk.Label(container, text="Ground Data File:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ground_entry = ttk.Entry(container, width=50)
        self.ground_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=self.load_ground_data).grid(row=1, column=2, padx=5, pady=5)

        # Output Directory
        ttk.Label(container, text="Output Directory:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.output_dir_entry = ttk.Entry(container, textvariable=self.output_dir_var, width=50)
        self.output_dir_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(container, text="Browse", command=self.browse_output_dir).grid(row=2, column=2, padx=5, pady=5)

        # Storm Threshold
        ttk.Label(container, text="Storm Threshold:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.threshold_mode_var = ttk.StringVar(value="Manual")  # New variable for threshold mode
        ttk.Radiobutton(container, text="Manual", variable=self.threshold_mode_var, value="Manual").grid(row=3, column=1, padx=5, pady=5, sticky="w")
        self.storm_entry = ttk.Entry(container, width=10)
        self.storm_entry.insert(0, str(DEFAULT_STORM_THRESHOLD))
        self.storm_entry.grid(row=3, column=2, padx=5, pady=5, sticky="w")
        ttk.Radiobutton(container, text="75th Percentile", variable=self.threshold_mode_var, value="Percentile").grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # Quartile Points
        ttk.Label(container, text="Quartile Points:").grid(row=5, column=0, padx=5, pady=5, sticky="w")
        self.quartile_var = ttk.StringVar(value="Quartiles")
        self.quartile_combobox = ttk.Combobox(container, textvariable=self.quartile_var, 
                                            values=list(QUARTILE_OPTIONS.keys()), state="readonly")
        self.quartile_combobox.grid(row=5, column=1, padx=5, pady=5, sticky="w")
        self.quartile_combobox.bind("<<ComboboxSelected>>", self.handle_quartile_selection)

        # Spline Method
        ttk.Label(container, text="Spline Method:").grid(row=6, column=0, padx=5, pady=5, sticky="w")
        self.spline_method_combobox = ttk.Combobox(container, textvariable=self.spline_method_var, 
                                                values=["Original", "Simplified"], state="readonly")
        self.spline_method_combobox.grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # Bias Correction Data Scope
        ttk.Label(container, text="Bias Correction Data Scope:").grid(row=7, column=0, padx=5, pady=5, sticky="w")
        self.bias_correction_scope_combobox = ttk.Combobox(container, textvariable=self.bias_correction_scope_var, 
                                                        values=["Common Dates", "Full Satellite"], state="readonly")
        self.bias_correction_scope_combobox.grid(row=7, column=1, padx=5, pady=5, sticky="w")

        # Aggregation Level
        ttk.Label(container, text="Aggregation Level:").grid(row=8, column=0, padx=5, pady=5, sticky="w")
        self.agg_level_var = ttk.StringVar(value="Daily")
        self.agg_level_combo = ttk.Combobox(
            container, textvariable=self.agg_level_var,
            values=["Daily", "Monthly", "Yearly"], state="readonly"
        )
        self.agg_level_combo.grid(row=8, column=1, padx=5, pady=5, sticky="w")

        # Aggregation Function
        ttk.Label(container, text="Aggregation Function:").grid(row=9, column=0, padx=5, pady=5, sticky="w")
        self.agg_func_var = ttk.StringVar(value="Mean")
        self.agg_func_combo = ttk.Combobox(
            container, textvariable=self.agg_func_var,
            values=["Mean", "Max", "Total"], state="readonly"
        )
        self.agg_func_combo.grid(row=9, column=1, padx=5, pady=5, sticky="w")

        # Checkbuttons
        self.zero_rain_check = ttk.Checkbutton(container, text="Enable Zero-Rain Adjustment", 
                                            variable=self.zero_rain_var)
        self.zero_rain_check.grid(row=10, column=0, columnspan=2, padx=5, pady=5, sticky="w")
        self.outlier_capping_check = ttk.Checkbutton(container, text="Enable Outlier Capping", 
                                                    variable=self.outlier_capping_var)
        self.outlier_capping_check.grid(row=10, column=2, padx=5, pady=5, sticky="w")

        # Buttons
        ttk.Button(container, text="Generate Synthetic Data", command=self.generate_synthetic_data).grid(row=11, column=0, columnspan=3, pady=10)
        ttk.Button(container, text="Run Analysis", command=self.run_analysis, style="primary.TButton").grid(row=12, column=0, columnspan=2, pady=10)
        ttk.Button(container, text="Reset Analysis", command=self.reset_analysis).grid(row=12, column=2, padx=5, pady=10)

        # Log Text Area with Scrollbar
        log_frame = ttk.Frame(container)
        log_frame.grid(row=13, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")
        self.log_text = ttk.Text(log_frame, height=10, width=80, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True)
        log_scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        # Dynamic timestamp for log message
        current_time = datetime.now().strftime("%I:%M %p %Z on %A, %B %d, %Y")
        self.log_text.insert("end", f"Application started at {current_time}.\n")

    def setup_results_tab(self):
        self.tree = ttk.Treeview(self.results_frame, columns=("Metric", "Original", "Linear", "Rank", "Spline"), show="headings")
        self.tree.heading("Metric", text="Metric")
        self.tree.heading("Original", text="Original")
        self.tree.heading("Linear", text="Linear")
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("Spline", text="Spline")
        self.tree.pack(pady=10, padx=10, fill="both", expand=True)
        ttk.Button(self.results_frame, text="Save Metrics", command=self.save_metrics).pack(pady=5)

    def setup_plots_tab(self):
        self.plot_var = ttk.StringVar()
        self.plot_combobox = ttk.Combobox(self.plots_frame, textvariable=self.plot_var, values=["Histogram", "CDF", "Time Series", "Metrics Comparison"])
        self.plot_combobox.set("Histogram")
        self.plot_combobox.pack(pady=5)
        self.plot_combobox.bind("<<ComboboxSelected>>", self.display_plot)

        ttk.Label(self.plots_frame, text="Time Series Start Date (YYYY-MM-DD):").pack(pady=5)
        self.start_date_entry = ttk.Entry(self.plots_frame, textvariable=self.start_date_var, width=20)
        self.start_date_entry.pack(pady=5)
        ttk.Label(self.plots_frame, text="Time Series End Date (YYYY-MM-DD):").pack(pady=5)
        self.end_date_entry = ttk.Entry(self.plots_frame, textvariable=self.end_date_var, width=20)
        self.end_date_entry.pack(pady=5)
        ttk.Button(self.plots_frame, text="Apply Date Range", command=self.apply_date_range).pack(pady=5)

        self.canvas_frame = ttk.Frame(self.plots_frame)
        self.canvas_frame.pack(pady=10, fill="both", expand=True)
        ttk.Button(self.plots_frame, text="Save Plot", command=self.save_plot).pack(pady=5)

    def reset_analysis(self):
        self.sat_entry.delete(0, "end")
        self.ground_entry.delete(0, "end")
        self.output_dir_var.set("")
        self.storm_entry.delete(0, "end")
        self.storm_entry.insert(0, str(DEFAULT_STORM_THRESHOLD))
        self.threshold_mode_var.set("Manual")  
        self.quartile_var.set("Quartiles")
        self.spline_method_var.set("Original")
        self.bias_correction_scope_var.set("Common Dates")
        self.agg_level_var.set("Daily")
        self.agg_func_var.set("Mean")
        self.zero_rain_var.set(False)
        self.outlier_capping_var.set(False)
        self.start_date_var.set("")
        self.end_date_var.set("")
        self.data = None 
        self.metrics = {}
        self.log_text.delete(1.0, "end")    
        current_time = datetime.now().strftime("%I:%M %p %Z on %A, %B %d, %Y")
        self.log_text.insert("end", f"Application reset at {current_time}.\n")

    def handle_quartile_selection(self, event):
        selection = self.quartile_var.get()
        if selection == "Custom":
            self.get_custom_quartiles()

    def get_custom_quartiles(self):
        top = Toplevel(self.root)
        top.title("Enter Custom Quartile Points")
        ttk.Label(top, text="Enter comma-separated quartile points (e.g., 25,50,75):").pack(pady=5)
        custom_entry = ttk.Entry(top, width=30)
        custom_entry.pack(pady=5)

        def confirm():
            try:
                points = [float(x.strip()) for x in custom_entry.get().split(",")]
                if not self.validate_quartile_points(points):
                    raise ValueError("Invalid quartile points")
                QUARTILE_OPTIONS["Custom"] = points
                self.log(f"Custom quartile points set: {points}")
                top.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Invalid input: {e}")
                self.log(f"Error setting custom quartile points: {e}")

        ttk.Button(top, text="Confirm", command=confirm).pack(pady=10)
        top.grab_set()

    def validate_quartile_points(self, points):
        if not points:
            return False
        if not all(0 <= p <= 100 for p in points):
            return False
        if not all(points[i] < points[i+1] for i in range(len(points)-1)):
            return False
        if len(set(points)) != len(points):
            return False
        return True

    def get_quartile_points(self):
        selection = self.quartile_var.get()
        points = QUARTILE_OPTIONS[selection]
        if points is None:
            raise ValueError("Custom quartile points not set")
        return points

    def log(self, message):
        self.log_text.insert("end", message + "\n")
        self.log_text.see("end")
        # Save log to file if log_file_path is set
        if self.log_file_path:
            try:
                with open(self.log_file_path, 'a') as f:
                    f.write(message + "\n")
            except Exception as e:
                self.log_text.insert("end", f"Error writing to log file: {e}\n")

    def select_columns(self, df, file_path, file_type):
        columns = list(df.columns)
        if not columns:
            raise ValueError("No columns found in file")

        time_col = [None]
        precip_col = [None]
        top = Toplevel(self.root)
        top.title(f"Select Columns for {file_type} Data")
        ttk.Label(top, text=f"Select columns for {file_path}:").pack(pady=5)

        ttk.Label(top, text="Time Column:").pack(pady=5)
        time_var = ttk.StringVar()
        time_combo = ttk.Combobox(top, textvariable=time_var, values=columns, state="readonly")
        time_combo.pack(pady=5)
        if columns:
            time_combo.set(columns[0])

        ttk.Label(top, text="Precipitation Column:").pack(pady=5)
        precip_var = ttk.StringVar()
        precip_combo = ttk.Combobox(top, textvariable=precip_var, values=columns, state="readonly")
        precip_combo.pack(pady=5)
        if len(columns) > 1:
            precip_combo.set(columns[1])

        def confirm():
            time_col[0] = time_var.get()
            precip_col[0] = precip_var.get()
            if not time_col[0] or not precip_col[0]:
                messagebox.showerror("Error", "Please select both time and precipitation columns")
                top.destroy()
                return
            if time_col[0] == precip_col[0]:
                messagebox.showerror("Error", "Time and precipitation columns must be different")
                top.destroy()
                return
            top.destroy()

        def cancel():
            time_col[0] = None
            precip_col[0] = None
            top.destroy()

        ttk.Button(top, text="Confirm", command=confirm).pack(side="left", padx=10, pady=10)
        ttk.Button(top, text="Cancel", command=cancel).pack(side="right", padx=10, pady=10)
        top.grab_set()
        self.root.wait_window(top)

        if not time_col[0] or not precip_col[0]:
            raise ValueError("Column selection cancelled or invalid")
        return time_col[0], precip_col[0]

    def aggregate_data(self, df, value_col):
        """Return daily, monthly, and yearly mean/max/total DataFrames for the given value_col."""
        df = df.copy()
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        # Daily (original, sorted)
        daily = df[['Date', value_col]].sort_values('Date').dropna(subset=[value_col])
        # Monthly
        monthly = df.groupby(['Year', 'Month'], as_index=False)[value_col].agg(['mean', 'max', 'sum'])
        monthly.columns = ['Year', 'Month', 'Mean', 'Max', 'Total']
        monthly['Date'] = pd.to_datetime(monthly['Year'].astype(str) + '-' + monthly['Month'].astype(str) + '-01')
        # Drop rows with NaN in any aggregation column
        monthly = monthly.dropna(subset=['Mean', 'Max', 'Total'])
        # Yearly
        yearly = df.groupby(['Year'], as_index=False)[value_col].agg(['mean', 'max', 'sum'])
        yearly.columns = ['Year', 'Mean', 'Max', 'Total']
        yearly['Date'] = pd.to_datetime(yearly['Year'].astype(str) + '-01-01')
        # Drop rows with NaN in any aggregation column
        yearly = yearly.dropna(subset=['Mean', 'Max', 'Total'])
        return daily, monthly[['Date', 'Mean', 'Max', 'Total']], yearly[['Date', 'Mean', 'Max', 'Total']]

    def load_sat_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel/CSV files", "*.xlsx *.csv")])
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_csv(file_path)
                time_col, precip_col = self.select_columns(df, file_path, "Satellite")
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if df[time_col].isna().any():
                    raise ValueError("Invalid or missing dates in time column")
                if not pd.to_numeric(df[precip_col], errors='coerce').notna().any():
                    raise ValueError("Precipitation column contains no valid numeric values")
                self.sat_df = df[[time_col, precip_col]].rename(columns={time_col: 'Date', precip_col: 'Satellite'})
                self.sat_entry.delete(0, "end")
                self.sat_entry.insert(0, file_path)
                self.log(f"Loaded satellite data: {file_path} (Time: {time_col}, Precipitation: {precip_col})")
                # Aggregation
                self.sat_daily, self.sat_monthly, self.sat_yearly = self.aggregate_data(self.sat_df, 'Satellite')
                self.log(f"Satellite aggregation: {len(self.sat_daily)} daily, {len(self.sat_monthly)} monthly, {len(self.sat_yearly)} yearly rows")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load satellite data: {e}")
                self.log(f"Error loading satellite data: {e}")

    def load_ground_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel/CSV files", "*.xlsx *.csv")])
        if file_path:
            try:
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                else:
                    df = pd.read_csv(file_path)
                time_col, precip_col = self.select_columns(df, file_path, "Ground")
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                if df[time_col].isna().any():
                    raise ValueError("Invalid or missing dates in time column")
                if not pd.to_numeric(df[precip_col], errors='coerce').notna().any():
                    raise ValueError("Precipitation column contains no valid numeric values")
                self.ground_df = df[[time_col, precip_col]].rename(columns={time_col: 'Date', precip_col: 'Ground'})
                self.ground_entry.delete(0, "end")
                self.ground_entry.insert(0, file_path)
                self.log(f"Loaded ground data: {file_path} (Time: {time_col}, Precipitation: {precip_col})")
                # Aggregation
                self.ground_daily, self.ground_monthly, self.ground_yearly = self.aggregate_data(self.ground_df, 'Ground')
                self.log(f"Ground aggregation: {len(self.ground_daily)} daily, {len(self.ground_monthly)} monthly, {len(self.ground_yearly)} yearly rows")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ground data: {e}")
                self.log(f"Error loading ground data: {e}")

    def generate_synthetic_data(self):
        n_sat, n_ground = 10000, 5000
        sat_dates = pd.date_range(start='2025-01-01', periods=n_sat, freq='D')
        ground_dates = pd.date_range(start='2025-01-01', periods=n_ground, freq='D')
        ground_zero_prob = 0.8
        satellite_zero_prob = 0.5
        ground_rainfall = np.clip(np.random.gamma(shape=2, scale=5, size=n_ground), 0, None)
        ground_rainfall[np.random.rand(n_ground) < ground_zero_prob] = 0
        satellite_rainfall = np.clip(np.random.gamma(shape=2, scale=5, size=n_sat) * 1.3 + np.random.normal(0, 2, n_sat), 0, None)
        satellite_rainfall[np.random.rand(n_sat) < satellite_zero_prob] = 0
        self.sat_df = pd.DataFrame({'Date': sat_dates, 'Satellite': satellite_rainfall})
        self.ground_df = pd.DataFrame({'Date': ground_dates, 'Ground': ground_rainfall})
        self.log(f"Generated synthetic data: {n_sat} satellite points ({satellite_zero_prob:.0%} dry), {n_ground} ground points ({ground_zero_prob:.0%} dry)")
        self.sat_entry.delete(0, "end")
        self.sat_entry.insert(0, "Synthetic Satellite Data")
        self.ground_entry.delete(0, "end")
        self.ground_entry.insert(0, "Synthetic Ground Data")

    def validate_data(self):
        if self.sat_df is None or self.ground_df is None:
            raise ValueError("Satellite or ground data not loaded")
        assert not self.sat_df['Date'].duplicated().any(), "Duplicate dates in satellite data"
        assert not self.ground_df['Date'].duplicated().any(), "Duplicate dates in ground data"
        assert self.sat_df['Date'].dtype == 'datetime64[ns]', "Satellite dates not in datetime64 format"
        assert self.ground_df['Date'].dtype == 'datetime64[ns]', "Ground dates not in datetime64 format"

        self.log(f"Satellite date range: {self.sat_df['Date'].min()} to {self.sat_df['Date'].max()}")
        self.log(f"Ground date range: {self.ground_df['Date'].min()} to {self.ground_df['Date'].max()}")
        self.log(f"Satellite stats: {len(self.sat_df)} days, mean={self.sat_df['Satellite'].mean():.2f} mm, std={self.sat_df['Satellite'].std():.2f} mm, zeros={(self.sat_df['Satellite'] == 0).mean():.2%}, NaNs={self.sat_df['Satellite'].isna().sum()}")
        self.log(f"Ground stats: {len(self.ground_df)} days, mean={self.ground_df['Ground'].mean():.2f} mm, std={self.ground_df['Ground'].std():.2f} mm, zeros={(self.ground_df['Ground'] == 0).mean():.2%}, NaNs={self.ground_df['Ground'].isna().sum()}")

        initial_sat_len = len(self.sat_df)
        initial_ground_len = len(self.ground_df)
        self.sat_df = self.sat_df.dropna(subset=['Satellite'])
        self.ground_df = self.ground_df.dropna(subset=['Ground'])
        self.log(f"Preprocess: Dropped {initial_sat_len - len(self.sat_df)} satellite rows and {initial_ground_len - len(self.ground_df)} ground rows with NaN values")

        common_dates = set(self.sat_df['Date']).intersection(set(self.ground_df['Date']))
        if len(common_dates) < 10:
            raise ValueError(f"Insufficient common dates ({len(common_dates)}) between satellite and ground data")
        self.log(f"Number of common dates: {len(common_dates)}")

        self.data = self.sat_df.merge(self.ground_df, on='Date', how='inner')
        if self.data.empty:
            raise ValueError("Merged dataset is empty after joining on common dates")
        self.log(f"Merged dataset shape: {self.data.shape}")

        ground_nonzero = self.data['Ground'][self.data['Ground'] > 0]
        sat_nonzero = self.data['Satellite'][self.data['Satellite'] > 0]
        if len(ground_nonzero) < 5 or len(sat_nonzero) < 5:
            self.log(f"Preprocess: Insufficient non-zero data - Ground: {len(ground_nonzero)}, Satellite: {len(sat_nonzero)}")
            raise ValueError("Insufficient non-zero data for analysis (minimum 5 non-zero points required)")

        ground_zero_frac = (self.data['Ground'] == 0).mean()
        sat_zero_frac = (self.data['Satellite'] == 0).mean()
        self.log(f"Initial dry day fractions (common dates): Ground {ground_zero_frac:.2%}, Satellite {sat_zero_frac:.2%}")
        if abs(ground_zero_frac - sat_zero_frac) > 0.1:
            self.log(f"Warning: Significant dry day disparity detected ({abs(ground_zero_frac - sat_zero_frac):.2%})")

        agg_level = self.agg_level_var.get()
        if agg_level in ["Monthly", "Yearly"]:
            self.log(f"Preprocess: Skipping zero-rain adjustment for {agg_level} aggregation")
        elif self.zero_rain_var.get():
            self.data['Satellite'] = self.adjust_zero_rain(self.data['Satellite'].values, self.data['Ground'].values, ground_zero_frac)
            new_sat_zero_frac = (self.data['Satellite'] == 0).mean()
            self.log(f"Preprocess: Adjusted satellite dry days to {new_sat_zero_frac:.2%} (target {ground_zero_frac:.2%})")
        else:
            self.log("Preprocess: Zero-rain adjustment disabled by user")

        if agg_level not in ["Monthly", "Yearly"]:
            sat_small = (self.data['Satellite'] < MIN_RAINFALL_THRESHOLD) & (self.data['Satellite'] > 0)
            ground_small = (self.data['Ground'] < MIN_RAINFALL_THRESHOLD) & (self.data['Ground'] > 0)
            self.data.loc[sat_small, 'Satellite'] = 0
            self.data.loc[ground_small, 'Ground'] = 0
            self.log(f"Preprocess: Satellite values < {MIN_RAINFALL_THRESHOLD} mm set to 0: {sat_small.sum()} points")
            self.log(f"Preprocess: Ground values < {MIN_RAINFALL_THRESHOLD} mm set to 0: {ground_small.sum()} points")

        if self.outlier_capping_var.get():
            sat_nonzero = self.data['Satellite'][self.data['Satellite'] > 0]
            ground_nonzero = self.data['Ground'][self.data['Ground'] > 0]
            if len(sat_nonzero) > 0:
                sat_max = np.percentile(sat_nonzero, 98)
                sat_outliers = self.data['Satellite'] > sat_max
                self.data.loc[sat_outliers, 'Satellite'] = sat_max
                self.log(f"Preprocess: Capped {sat_outliers.sum()} satellite outliers at {sat_max:.2f} mm")
            if len(ground_nonzero) > 0:
                ground_max = np.percentile(ground_nonzero, 98)
                ground_outliers = self.data['Ground'] > ground_max
                self.data.loc[ground_outliers, 'Ground'] = ground_max
                self.log(f"Preprocess: Capped {ground_outliers.sum()} ground outliers at {ground_max:.2f} mm")
        else:
            self.log("Preprocess: Outlier capping disabled by user")

        self.log(f"Final data stats (common dates) - Ground: mean={self.data['Ground'].mean():.2f} mm, std={self.data['Ground'].std():.2f} mm, zeros={(self.data['Ground'] == 0).mean():.2%}")
        self.log(f"Final data stats (common dates) - Satellite: mean={self.data['Satellite'].mean():.2f} mm, std={self.data['Satellite'].std():.2f} mm, zeros={(self.data['Satellite'] == 0).mean():.2%}")

        assert set(self.data['Date']).issubset(set(self.sat_df['Date'])), "Merged dates not in satellite data"
        assert set(self.data['Date']).issubset(set(self.ground_df['Date'])), "Merged dates not in ground data"

    def predict_rain(self, satellite_data, ground_data):
        ground_zero_frac = np.mean(ground_data == 0)
        sat_zero_frac = np.mean(satellite_data == 0)
        self.log(f"Preprocess: Initial stats - Satellite: mean={satellite_data.mean():.2f}, std={satellite_data.std():.2f}, zeros={sat_zero_frac:.2%}")
        self.log(f"Preprocess: Ground: mean={ground_data.mean():.2f}, std={ground_data.std():.2f}, zeros={ground_zero_frac:.2%}")
        if abs(ground_zero_frac - sat_zero_frac) > 0.01:
            self.log(f"Warning: Significant dry day disparity: Ground {ground_zero_frac:.2%}, Satellite {sat_zero_frac:.2%}")

        agg_level = self.agg_level_var.get()
        if agg_level in ["Monthly", "Yearly"]:
            self.log(f"Preprocess: Skipping zero-rain adjustment in predict_rain for {agg_level} aggregation")
            rain_mask = satellite_data > 0
            sat_zero_frac = np.mean(~rain_mask)
            self.log(f"Preprocess: Using non-zero satellite data as rain mask, zero-rain fraction: {sat_zero_frac:.2%}")
        elif np.any(satellite_data > 0):
            sat_nonzero = satellite_data[satellite_data > 0]
            if len(sat_nonzero) > 10:
                try:
                    # Direct thresholding
                    threshold = np.percentile(sat_nonzero, (1 - ground_zero_frac) * 100) if np.sum(sat_nonzero > 0) > 0 else 0
                    rain_mask = satellite_data >= threshold
                    sat_zero_frac = np.mean(~rain_mask)
                    self.log(f"Preprocess: Direct thresholding succeeded, zero-rain fraction: {sat_zero_frac:.2%}, threshold: {threshold:.3f} mm")
                except Exception as e:
                    self.log(f"Preprocess: Direct thresholding failed ({e}), using iterative thresholding")
                    sorted_sat = np.sort(sat_nonzero)
                    target_zero_count = int(ground_zero_frac * len(satellite_data))
                    current_zero_count = np.sum(satellite_data == 0)
                    remaining_zeros = target_zero_count - current_zero_count
                    if remaining_zeros > 0 and len(sorted_sat) > 0:
                        max_threshold = np.percentile(sorted_sat, 99)
                        low, high = sorted_sat[0], min(sorted_sat[-1], max_threshold)
                        threshold = low
                        for iteration in range(50):
                            threshold = (low + high) / 2
                            rain_mask = satellite_data >= threshold
                            new_zero_count = np.sum(~rain_mask)
                            error = (new_zero_count - target_zero_count) / len(satellite_data)
                            if abs(error) < 0.005 or high - low < 1e-3:
                                break
                            if new_zero_count < target_zero_count:
                                high = threshold
                            else:
                                low = threshold
                        rain_mask = satellite_data >= threshold
                        sat_zero_frac = np.mean(~rain_mask)
                        self.log(f"Preprocess: Iterative threshold set to {threshold:.3f} mm after {iteration+1} iterations, zero-rain fraction: {sat_zero_frac:.2%}")
                    else:
                        rain_mask = satellite_data > 0
                        sat_zero_frac = np.mean(~rain_mask)
                        self.log(f"Preprocess: No adjustment needed or insufficient non-zero data, zero-rain fraction: {sat_zero_frac:.2%}")
            else:
                rain_mask = satellite_data > 0
                sat_zero_frac = np.mean(~rain_mask)
                self.log("Preprocess: Insufficient non-zero satellite data")
        else:
            rain_mask = np.zeros_like(satellite_data, dtype=bool)
            sat_zero_frac = 1.0
            self.log("Preprocess: No non-zero satellite data")

        non_zero_count = np.sum(rain_mask)
        self.log(f"Preprocess: Adjusted satellite zero-rain fraction: {sat_zero_frac:.2%} (target: {ground_zero_frac:.2%}), Non-zero count: {non_zero_count}")
        return rain_mask
    
    def adjust_zero_rain(self, satellite_data, ground_data, target_zero_frac):
        if np.sum(satellite_data > 0) < 5:
            self.log("Too few non-zero satellite data points (< 5); skipping zero-rain adjustment")
            return satellite_data
        current_zero_frac = np.mean(satellite_data == 0)
        if abs(current_zero_frac - target_zero_frac) > 0.01:
            sat_nonzero = satellite_data[satellite_data > 0]
            sorted_sat = np.sort(sat_nonzero)
            target_zero_count = int(target_zero_frac * len(satellite_data))
            current_zero_count = np.sum(satellite_data == 0)
            additional_zeros = target_zero_count - current_zero_count
            if additional_zeros > 0:
                threshold_idx = int((1 - target_zero_frac) * len(sorted_sat))
                threshold = sorted_sat[threshold_idx] if threshold_idx < len(sorted_sat) else sorted_sat[-1]
                adjusted = satellite_data.copy()
                adjusted[adjusted <= threshold] = 0
                new_zero_frac = np.mean(adjusted == 0)
                self.log(f"Adjusted satellite zero-rain to {new_zero_frac:.2%} with threshold {threshold:.3f} mm")
                return adjusted
            elif additional_zeros < 0:
                self.log("Current zero fraction higher than target; no adjustment needed")
                return satellite_data
        self.log("Zero-rain fraction within 1% of target; no adjustment needed")
        return satellite_data

    # Original spline method with adaptive knot selection
    def spline_quantile_mapping_original(self, satellite_data, ground_data):
        # Clean non-finite values
        valid_mask = np.isfinite(satellite_data) & np.isfinite(ground_data)
        if not np.all(valid_mask):
            self.log(f"Spline: Dropped {np.sum(~valid_mask)} non-finite values")
            satellite_data = satellite_data[valid_mask].copy()
            ground_data = ground_data[valid_mask].copy()

        if len(satellite_data) == 0 or len(ground_data) == 0:
            self.log("Spline: No data after cleaning; returning zeros")
            self.store_diagnostic("spline", {"n_knots": 0, "zero_frac": 1.0})
            return np.zeros_like(valid_mask, dtype=float)

        # Predict rain events
        rain_mask = self.predict_rain(satellite_data, ground_data)
        satellite_nonzero = satellite_data[rain_mask]
        ground_nonzero = ground_data[ground_data > 0]
        self.log(f"Spline: Non-zero counts - satellite={len(satellite_nonzero)}, ground={len(ground_nonzero)}")

        # Check for sufficient non-zero data
        agg_level = self.agg_level_var.get()
        min_nonzero = 5 if agg_level in ["Monthly", "Yearly"] else 20
        if len(satellite_nonzero) < min_nonzero or len(ground_nonzero) < min_nonzero:
            self.log(f"Spline: Insufficient non-zero data (min={min_nonzero}); falling back to linear")
            quartile_points = np.linspace(1, 100, 100)
            sat_quantiles = np.percentile(satellite_nonzero, quartile_points) if len(satellite_nonzero) > 0 else np.zeros_like(quartile_points)
            ground_quantiles = np.percentile(ground_nonzero, quartile_points) if len(ground_nonzero) > 0 else np.zeros_like(quartile_points)
            matched = self.linear_quantile_mapping(satellite_data, sat_quantiles, ground_quantiles, quartile_points)
            self.store_diagnostic("spline", {"n_knots": 0, "zero_frac": np.mean(matched == 0)})
            return matched

        # Adjust knot selection based on aggregation level
        n = len(satellite_nonzero)
        if agg_level in ["Monthly", "Yearly"]:
            n_knots = max(3, min(5, int(np.log2(n) + 1)))  # Fewer knots for monthly/yearly
            self.log(f"Spline: {agg_level} data detected; using {n_knots} knots")
        else:
            n_knots = max(5, min(15, int(np.log2(n) + 1)))  # Default for daily
            self.log(f"Spline: Daily data detected; using {n_knots} knots")

        try:
            # Dynamic bins for knot candidates
            sat_iqr = iqr(satellite_nonzero, nan_policy='omit')
            ground_iqr = iqr(ground_nonzero, nan_policy='omit')
            sat_h = 2 * sat_iqr * n ** (-1/3) if sat_iqr > 0 else np.std(satellite_nonzero) / 10
            ground_h = 2 * ground_iqr * n ** (-1/3) if ground_iqr > 0 else np.std(ground_nonzero) / 10
            sat_bins = int(np.clip(np.ceil((np.max(satellite_nonzero) - np.min(satellite_nonzero)) / sat_h), 5, 15 if agg_level in ["Monthly", "Yearly"] else 30)) if sat_h > 0 else 10
            ground_bins = int(np.clip(np.ceil((np.max(ground_nonzero) - np.min(ground_nonzero)) / ground_h), 5, 15 if agg_level in ["Monthly", "Yearly"] else 30)) if ground_h > 0 else 10
            sat_hist, sat_edges = np.histogram(satellite_nonzero, bins=sat_bins, density=True)
            ground_hist, ground_edges = np.histogram(ground_nonzero, bins=ground_bins, density=True)
            sat_dense = sat_edges[:-1][sat_hist > 0.1 * np.max(sat_hist)]
            ground_dense = ground_edges[:-1][ground_hist > 0.1 * np.max(ground_hist)]
            self.log(f"Spline: Bins - satellite={sat_bins}, ground={ground_bins}")

            # Knot selection
            percentiles = [5, 50, 95] if agg_level in ["Monthly", "Yearly"] else [5, 10, 25, 50, 75, 90, 95]
            sat_percentiles = np.percentile(satellite_nonzero, percentiles)
            ground_percentiles = np.percentile(ground_nonzero, percentiles)
            knot_candidates = np.unique(np.concatenate([sat_dense, ground_dense, sat_percentiles, ground_percentiles]))
            knots = np.linspace(knot_candidates.min(), knot_candidates.max(), n_knots)
            self.log(f"Spline: Knots selected: {knots.tolist()}")

            # Align non-zero data
            min_nonzero = min(len(satellite_nonzero), len(ground_nonzero))
            sat_sorted = np.sort(satellite_nonzero)[:min_nonzero]
            ground_sorted = np.sort(ground_nonzero)[:min_nonzero]
            quantiles = np.linspace(0, 1, min_nonzero)
            sat_quantiles = np.interp(quantiles, np.linspace(0, 1, min_nonzero), sat_sorted)
            ground_quantiles = np.interp(quantiles, np.linspace(0, 1, min_nonzero), ground_sorted)

            # Fit spline
            sat_quantiles = sat_quantiles + np.arange(len(sat_quantiles)) * 1e-10  # Avoid duplicate x-values
            spl = CubicSpline(sat_quantiles, ground_quantiles, bc_type='natural')
            matched = np.zeros_like(satellite_data, dtype=float)
            matched[rain_mask] = np.clip(spl(satellite_data[rain_mask]), 0, np.max(ground_nonzero))

            # Skip zero-rain adjustment and small value zeroing for monthly/yearly
            if agg_level not in ["Monthly", "Yearly"]:
                # Zero-rain adjustment
                ground_zero_frac = np.mean(ground_data == 0)
                matched_zero_frac = np.mean(matched == 0)
                if abs(ground_zero_frac - matched_zero_frac) > 0.01:
                    threshold = np.percentile(matched[matched > 0], (1 - ground_zero_frac) * 100) if np.sum(matched > 0) > 0 else 0
                    matched[matched <= threshold] = 0
                    self.log(f"Spline: Adjusted zero-rain to {ground_zero_frac:.2%} with threshold {threshold:.3f} mm")

                # Set small values to zero
                zeros_below_1mm = np.sum((matched > 0) & (matched < 1.0))
                matched[(matched > 0) & (matched < 1.0)] = 0
                self.log(f"Spline: Values < 1 mm set to 0: {zeros_below_1mm} points")

            # Outlier capping
            if self.outlier_capping_var.get():
                sat_max = np.percentile(satellite_nonzero, 98) if len(satellite_nonzero) > 0 else np.max(satellite_data)
                capped = np.sum(matched[rain_mask] > sat_max)
                matched[rain_mask] = np.clip(matched[rain_mask], 0, sat_max)
                self.log(f"Spline: Capped {capped} points to {sat_max:.2f} mm")

            # Diagnostics
            self.store_diagnostic("spline", {"n_knots": n_knots, "zero_frac": np.mean(matched == 0)})
            return matched

        except Exception as e:
            self.log(f"Spline: Error: {str(e)}; falling back to linear")
            quartile_points = np.linspace(1, 100, 100)
            sat_quantiles = np.percentile(satellite_nonzero, quartile_points) if len(satellite_nonzero) > 0 else np.zeros_like(quartile_points)
            ground_quantiles = np.percentile(ground_nonzero, quartile_points) if len(ground_nonzero) > 0 else np.zeros_like(quartile_points)
            matched = self.linear_quantile_mapping(satellite_data, sat_quantiles, ground_quantiles, quartile_points)
            self.store_diagnostic("spline", {"n_knots": 0, "zero_frac": np.mean(matched == 0)})
            return matched

    def spline_quantile_mapping_simplified(self, satellite_data, ground_data):
        # Clean input data for non-finite values
        sat_finite = np.isfinite(satellite_data)
        ground_finite = np.isfinite(ground_data)
        if not np.all(sat_finite) or not np.all(ground_finite):
            self.log(f"Spline (Simplified): Warning - Found {np.sum(~sat_finite)} non-finite satellite values and {np.sum(~ground_finite)} non-finite ground values")
            satellite_data = satellite_data[sat_finite].copy()
            ground_data = ground_data[ground_finite].copy()
            if len(satellite_data) == 0 or len(ground_data) == 0:
                self.log("Spline (Simplified): Error - No finite data remaining after cleaning; returning zeros")
                self.store_diagnostic("spline", {
                    "n_knots": 0,
                    "zero_frac": 1.0,
                    "method": "failed",
                    "ks_statistic": np.nan
                })
                return np.zeros_like(sat_finite, dtype=float)
        # Log input data statistics
        sat_zero_frac = np.mean(satellite_data == 0)
        ground_zero_frac = np.mean(ground_data == 0)
        self.log(f"Spline (Simplified): Satellite stats - mean={satellite_data.mean():.2f} mm, std={satellite_data.std():.2f} mm, zeros={sat_zero_frac:.2%}")
        self.log(f"Spline (Simplified): Ground stats - mean={ground_data.mean():.2f} mm, std={ground_data.std():.2f} mm, zeros={ground_zero_frac:.2%}")

        # Compute ranks and percentiles for the full satellite dataset
        original_indices = np.argsort(np.argsort(satellite_data))
        n_sat = len(satellite_data)
        n_ground = len(ground_data)
        if n_sat == 0 or n_ground == 0:
            self.log(f"Spline (Simplified): Empty satellite or ground data. n_sat={n_sat}, n_ground={n_ground}")
            return np.zeros_like(satellite_data)
        percentiles = original_indices / (n_sat - 1) if n_sat > 1 else np.zeros_like(original_indices)
        ground_sorted = np.sort(ground_data)
        ground_percentiles = np.linspace(0, 1, n_ground)
        unique_ground = np.unique(ground_sorted)
        if len(unique_ground) < len(ground_sorted):
            self.log(f"Spline (Simplified): Removed {len(ground_sorted) - len(unique_ground)} duplicate values from ground data")
        try:
            spline = CubicSpline(ground_percentiles, ground_sorted, bc_type='natural')
            ground_values = spline(percentiles)
            ground_values = np.clip(ground_values, 0, None)
            self.log(f"Spline (Simplified): Successfully fitted cubic spline with {n_ground} points")
            method = "cubic_spline"
        except Exception as e:
            self.log(f"Spline (Simplified): Warning - Spline failed ({e}); using linear interpolation")
            finite_mask = np.isfinite(ground_sorted)
            if not np.any(finite_mask):
                self.log("Spline (Simplified): Error - No finite ground data for interpolation; returning zeros")
                self.store_diagnostic("spline", {
                    "n_knots": 0,
                    "zero_frac": 1.0,
                    "method": "failed",
                    "ks_statistic": np.nan
                })
                return np.zeros_like(satellite_data, dtype=float)
            ground_sorted_finite = ground_sorted[finite_mask]
            ground_percentiles_finite = ground_percentiles[finite_mask]
            try:
                ground_values = np.interp(percentiles, ground_percentiles_finite, ground_sorted_finite)
            except Exception as e2:
                self.log(f"Spline (Simplified): Linear interpolation failed: {e2}. percentiles: {percentiles}, ground_percentiles_finite: {ground_percentiles_finite}, ground_sorted_finite: {ground_sorted_finite}")
                return np.zeros_like(satellite_data, dtype=float)
            ground_values = np.clip(ground_values, 0, None)
            method = "linear_interpolation"
        corrected = np.zeros_like(satellite_data, dtype=float)
        try:
            for i, idx in enumerate(original_indices):
                if idx < len(ground_values):
                    corrected[i] = ground_values[idx]
                else:
                    self.log(f"Spline (Simplified): Index {idx} out of bounds for ground_values of length {len(ground_values)}. Setting to 0.")
                    corrected[i] = 0
        except Exception as e:
            self.log(f"Spline (Simplified): Error assigning corrected values: {e}. original_indices: {original_indices}, ground_values: {ground_values}")
            corrected[:] = 0
        if not np.all(np.isfinite(corrected)):
            self.log(f"Spline (Simplified): Warning - Corrected data contains {np.sum(~np.isfinite(corrected))} non-finite values; setting to 0")
            corrected[~np.isfinite(corrected)] = 0
        corrected_zero_frac = np.mean(corrected == 0)
        self.log(f"Spline (Simplified): Corrected stats - mean={corrected.mean():.2f} mm, std={corrected.std():.2f} mm, zeros={corrected_zero_frac:.2%}")
        ks_stat, _ = ks_2samp(ground_data, corrected) if len(ground_data) > 0 and len(corrected) > 0 else (np.nan, np.nan)
        self.log(f"Spline (Simplified): KS statistic (corrected vs. ground): {ks_stat:.3f} (lower is better)")
        status = "Good" if ks_stat < 0.1 and np.isfinite(ks_stat) else "Check required"
        self.log(f"Spline (Simplified): Distribution status: {status}")
        self.store_diagnostic("spline", {
            "n_knots": n_ground if method == "cubic_spline" else 0,
            "zero_frac": float(corrected_zero_frac),
            "method": method,
            "ks_statistic": float(ks_stat),
            "sat_mean": float(satellite_data.mean()),
            "sat_std": float(satellite_data.std()),
            "sat_zero_frac": float(np.mean(satellite_data == 0)),
            "ground_mean": float(ground_data.mean()),
            "ground_std": float(ground_data.std()),
            "ground_zero_frac": float(np.mean(ground_data == 0)),
            "corrected_mean": float(corrected.mean()),
            "corrected_std": float(corrected.std())
        })
        return corrected

    # Wrapper to select spline method based on user input
    def spline_quantile_mapping(self, satellite_data, ground_data):
        spline_method = self.spline_method_var.get()
        self.log(f"Using spline method: {spline_method}")
        if spline_method == "Original":
            return self.spline_quantile_mapping_original(satellite_data, ground_data)
        else:  # Simplified
            return self.spline_quantile_mapping_simplified(satellite_data, ground_data)

    def rank_based_mapping(self, sat_data, ground_data, return_ordered=True):
        if len(sat_data) < 5 or len(ground_data) < 5:
            self.log(f"Rank: Dataset too small (sat={len(sat_data)}, ground={len(ground_data)}); returning zeros")
            return np.zeros_like(sat_data)

        rain_mask = self.predict_rain(sat_data, ground_data)
        sat_zero_frac = np.mean(~rain_mask)
        self.log(f"Rank: Initial satellite zero fraction: {sat_zero_frac:.2%}")
        agg_level = self.agg_level_var.get()
        if sat_zero_frac > 0.999 or (sat_zero_frac < 0.001 and agg_level not in ["Monthly", "Yearly"]):
            self.log(f"Rank: Extreme zero-rain fraction ({sat_zero_frac:.2%}), returning zeros")
            self.store_diagnostic("rank", {"zero_frac": 1.0, "rank_correlation": 1.0, "mapping_method": "None"})
            return np.zeros_like(sat_data)

        sat_nonzero = sat_data[rain_mask]
        ground_nonzero = ground_data[ground_data > 0]
        self.log(f"Rank: Non-zero counts - satellite={len(sat_nonzero)}, ground={len(ground_nonzero)}")
        
        if agg_level == "Yearly":
            min_nonzero = 5
        elif agg_level == "Monthly":
            min_nonzero = 10
        else:
            min_nonzero = 50
        if len(sat_nonzero) < min_nonzero or len(ground_nonzero) < min_nonzero:
            self.log(f"Rank: Insufficient non-zero data (min={min_nonzero}, sat={len(sat_nonzero)}, ground={len(ground_nonzero)}); using linear")
            quartile_points = np.linspace(1, 100, max(2, min(100, len(sat_nonzero))))
            sat_quantiles = np.percentile(sat_nonzero, quartile_points) if len(sat_nonzero) > 0 else np.zeros_like(quartile_points)
            ground_quantiles = np.percentile(ground_nonzero, quartile_points) if len(ground_nonzero) > 0 else np.zeros_like(quartile_points)
            matched = self.linear_quantile_mapping(sat_data, sat_quantiles, ground_quantiles, quartile_points)
            self.store_diagnostic("rank", {"zero_frac": np.mean(matched == 0), "rank_correlation": 1.0, "mapping_method": "Linear"})
            return matched

        try:
            min_nonzero = min(len(sat_nonzero), len(ground_nonzero))
            sat_nonzero = sat_nonzero[:min_nonzero]
            ground_nonzero = ground_nonzero[:min_nonzero]
            sat_ranks = rankdata(sat_nonzero, method='average') - 1
            n_sat = len(sat_nonzero)
            n_ground = len(ground_nonzero)
            percentiles = sat_ranks / n_sat
            sorted_ground = np.sort(ground_nonzero)
            indices = np.round(percentiles * (n_ground - 1)).astype(int)
            ground_values = sorted_ground[indices]
            mapping_method = "Nearest-neighbor"
            self.log("Rank: Used nearest-neighbor mapping")
        except (ValueError, RuntimeError) as e:
            self.log(f"Rank: Mapping failed ({e}), using linear interpolation")
            percentiles = sat_ranks / n_sat
            ground_values = np.interp(percentiles, np.linspace(0, 1, n_ground), np.sort(ground_nonzero))
            mapping_method = "Linear interpolation"

        ground_values = np.clip(ground_values, 0, np.max(ground_nonzero))
        corrected = np.zeros_like(sat_data)
        nonzero_indices = np.where(rain_mask)[0][:min_nonzero]
        corrected[nonzero_indices] = ground_values
        if agg_level not in ["Monthly", "Yearly"]:
            corrected[corrected < 1.0] = 0
            self.log(f"Rank: Set {np.sum((corrected > 0) & (corrected < 1.0))} values < 1.0 mm to 0")

        if self.outlier_capping_var.get():
            sat_max = np.percentile(sat_nonzero, 98) if len(sat_nonzero) > 0 else np.max(sat_data)
            capped = np.sum(corrected > sat_max)
            corrected = np.clip(corrected, 0, sat_max)
            self.log(f"Rank: Capped {capped} points to {sat_max:.2f} mm")

        final_ranks = rankdata(ground_values, method='average')
        rank_correlation = np.corrcoef(sat_ranks, final_ranks)[0, 1] if len(sat_ranks) > 1 else 1.0
        self.log(f"Rank: Correlation: {rank_correlation:.3f}")
        self.store_diagnostic("rank", {
            "zero_frac": np.mean(corrected == 0),
            "rank_correlation": rank_correlation,
            "mapping_method": mapping_method
        })
        return corrected if return_ordered else ground_values

    def evaluate_correction_methods(self, ground_data, sat_linear, sat_rank, sat_spline):
        log_lines = []
        ground_zero_frac = ground_data.eq(0).mean()
        linear_zero_frac = sat_linear.eq(0).mean()
        rank_zero_frac = self.get_diagnostic("rank", "zero_frac", sat_rank.eq(0).mean())
        spline_zero_frac = self.get_diagnostic("spline", "zero_frac", sat_spline.eq(0).mean())
        
        log_lines.append("Dry Day Fraction Alignment Check:")
        log_lines.append(f"  - Ground: {ground_zero_frac:.2%}")
        log_lines.append(f"  - Linear: {linear_zero_frac:.2%} (diff: {abs(linear_zero_frac - ground_zero_frac):.2%})")
        log_lines.append(f"  - Rank: {rank_zero_frac:.2%} (diff: {abs(rank_zero_frac - ground_zero_frac):.2%})")
        log_lines.append(f"  - Spline: {spline_zero_frac:.2%} (diff: {abs(spline_zero_frac - ground_zero_frac):.2%})")
        log_lines.append(f"  - Status: {'Good' if max(abs(linear_zero_frac - ground_zero_frac), abs(rank_zero_frac - ground_zero_frac), abs(spline_zero_frac - ground_zero_frac)) < 0.05 else 'Check required'}")
        
        sat_quantiles = np.percentile(sat_linear[sat_linear > 0], self.quartile_points)
        ground_quantiles = np.percentile(ground_data[ground_data > 0], self.quartile_points)
        quantile_diffs = np.abs(sat_quantiles - ground_quantiles)
        ks_stat_linear, _ = ks_2samp(ground_data.dropna(), sat_linear.dropna()) if len(ground_data.dropna()) > 0 and len(sat_linear.dropna()) > 0 else (np.nan, np.nan)
        log_lines.append("Linear Quantile mapping Evaluation:")
        log_lines.append(f"  - Quantile differences at {self.quartile_points}: {quantile_diffs.round(3).tolist()} mm")
        log_lines.append(f"  - Non-finite values: {np.sum(np.isnan(quantile_diffs)) + np.sum(np.isinf(quantile_diffs))} (should be 0)")
        log_lines.append(f"  - KS statistic: {ks_stat_linear:.3f} (lower is better)")
        log_lines.append(f"  - Status: {'Good' if ks_stat_linear < 0.1 else 'Check required'}")
        
        ks_stat_rank, _ = ks_2samp(ground_data[ground_data > 0], sat_rank[sat_rank > 0]) if len(ground_data[ground_data > 0]) > 0 and len(sat_rank[sat_rank > 0]) > 0 else (np.nan, np.nan)
        log_lines.append("Rank-Based mapping Evaluation:")
        log_lines.append(f"  - Rank preservation: {'Preserved' if self.get_diagnostic('rank', 'rank_correlation', 1.0) > 0.95 else 'Not preserved'}")
        log_lines.append(f"  - Rank correlation: {self.get_diagnostic('rank', 'rank_correlation', 1.0):.3f}")
        log_lines.append(f"  - Mapping method: {self.get_diagnostic('rank', 'mapping_method', 'Nearest-neighbor')}")
        log_lines.append(f"  - KS statistic (non-zero): {ks_stat_rank:.3f} (lower is better)")
        log_lines.append(f"  - Status: {'Good' if ks_stat_rank < 0.1 else 'Check required'}")
        
        n_knots = self.get_diagnostic("spline", "n_knots", 0)
        ks_stat_spline, _ = ks_2samp(ground_data.dropna(), sat_spline.dropna()) if len(ground_data.dropna()) > 0 and len(sat_spline.dropna()) > 0 else (np.nan, np.nan)
        log_lines.append(f"Cubic Spline ({self.spline_method_var.get()}) Quantile mapping Evaluation:")
        log_lines.append(f"  - Spline status: {'Used' if n_knots >= 5 else 'Fallback to linear interpolation'}")
        log_lines.append(f"  - Number of knots: {n_knots if n_knots >= 5 else 'N/A'}")
        log_lines.append(f"  - KS statistic: {ks_stat_spline:.3f} (lower is better)")
        log_lines.append(f"  - Status: {'Good' if n_knots >= 5 and (ks_stat_spline < 0.1 if not np.isnan(ks_stat_spline) else False) else 'Check required'}")
        
        return log_lines

    def apply_date_range(self):
        try:
            start_date = pd.to_datetime(self.start_date_var.get())
            end_date = pd.to_datetime(self.end_date_var.get())
            if self.data is None:
                raise ValueError("No data loaded")
            min_date = self.data['Date'].min()
            max_date = self.data['Date'].max()
            if start_date < min_date or end_date > max_date or start_date > end_date:
                raise ValueError(f"Date range must be between {min_date.date()} and {max_date.date()} and start date must be before end date")
            self.log(f"Applied time series date range: {start_date.date()} to {end_date.date()}")
            self.generate_plots(start_date=start_date, end_date=end_date, save=False) 
            self.display_plot(None)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid date range: {e}")
            self.log(f"Error applying date range: {e}")

    def get_selected_agg_data(self, is_sat=True):
        level = self.agg_level_var.get()
        func = self.agg_func_var.get()
        col = {"Mean": "Mean", "Max": "Max", "Total": "Total"}.get(func, "Mean")
        if is_sat:
            if level == "Daily":
                data = self.sat_daily.rename(columns={"Satellite": "Value"}).dropna(subset=["Value"])
            elif level == "Monthly":
                data = self.sat_monthly[["Date", col]].rename(columns={col: "Value"}).dropna(subset=["Value"])
            elif level == "Yearly":
                data = self.sat_yearly[["Date", col]].rename(columns={col: "Value"}).dropna(subset=["Value"])
            else:
                self.log(f"Invalid aggregation level: {level}")
                return None
        else:
            if level == "Daily":
                data = self.ground_daily.rename(columns={"Ground": "Value"}).dropna(subset=["Value"])
            elif level == "Monthly":
                data = self.ground_monthly[["Date", col]].rename(columns={col: "Value"}).dropna(subset=["Value"])
            elif level == "Yearly":
                data = self.ground_yearly[["Date", col]].rename(columns={col: "Value"}).dropna(subset=["Value"])
            else:
                self.log(f"Invalid aggregation level: {level}")
                return None
        self.log(f"Selected {level} {func} data ({'Satellite' if is_sat else 'Ground'}): {len(data)} rows, {data['Value'].isna().sum()} NaNs, {data['Value'].eq(0).sum()} zeros")
        return data
    
    def compute_75th_percentile_threshold(self, ground_data, agg_level):
        ground_data = ground_data[~np.isnan(ground_data)]  # Remove NaNs
        if len(ground_data) < 5:
            self.log("Warning: Too few ground data points (<5), using default threshold 20.0 mm")
            return 20.0
        if agg_level in ["Daily", "Monthly"]:
            non_zero_ground = ground_data[ground_data > 0]
            if len(non_zero_ground) > 0:
                threshold = np.percentile(non_zero_ground, 75)
                self.log(f"Computed 75th percentile (non-zero): {threshold:.2f} mm")
            else:
                self.log("Warning: No non-zero ground data, using default threshold 20.0 mm")
                threshold = 20.0
        else:  # Yearly
            threshold = np.percentile(ground_data, 75)
            self.log(f"Computed 75th percentile: {threshold:.2f} mm")
        return round(threshold, 2)


    def run_analysis(self):
        try:
            output_capture = StringIO()
            sys.stdout = output_capture
            
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            self.log_file_path = os.path.join(self.output_dir_var.get(), f"log_{timestamp}.txt")
            self.log(f"Starting analysis. Logs will be saved to {self.log_file_path}")

            # Validate and prepare data first
            self.validate_data()
            sat_agg = self.get_selected_agg_data(is_sat=True)
            ground_agg = self.get_selected_agg_data(is_sat=False)
            
            if sat_agg is None or ground_agg is None:
                raise ValueError("Failed to retrieve aggregated data")
            
            # Log data before merge
            self.log(f"Satellite agg stats: {len(sat_agg)} rows, mean={sat_agg['Value'].mean():.2f}, NaNs={sat_agg['Value'].isna().sum()}, zeros={(sat_agg['Value'] == 0).sum()}")
            self.log(f"Ground agg stats: {len(ground_agg)} rows, mean={ground_agg['Value'].mean():.2f}, NaNs={ground_agg['Value'].isna().sum()}, zeros={(ground_agg['Value'] == 0).sum()}")
            
            merged = pd.merge(sat_agg, ground_agg, on='Date', how='inner', suffixes=('_sat', '_ground'))
            self.log(f"Merged aggregation shape: {merged.shape}\nHead:\n{merged.head()}\n")
            
            # Check merged data validity
            if merged.empty:
                self.log("Error: Merged satellite and ground data is empty after aggregation/merge. Check input files, aggregation level, and date overlap.")
                messagebox.showerror("Error", "Merged satellite and ground data is empty after aggregation/merge. Check input files, aggregation level, and date overlap.")
                return
            
            # Validate column names
            expected_cols = ['Date', 'Value_sat', 'Value_ground']
            if not all(col in merged.columns for col in expected_cols):
                self.log(f"Error: Merged DataFrame missing expected columns. Found: {merged.columns.tolist()}")
                raise ValueError(f"Merged DataFrame missing expected columns: {expected_cols}")
            
            initial_len = len(merged)
            merged = merged.dropna(subset=['Value_sat', 'Value_ground'])
            self.log(f"Preprocess: Dropped {initial_len - len(merged)} rows with NaN values in merged data")
            
            # Log zeros in merged data
            sat_zeros = (merged['Value_sat'] == 0).sum()
            ground_zeros = (merged['Value_ground'] == 0).sum()
            self.log(f"Merged data zeros: Satellite {sat_zeros}/{len(merged)} ({sat_zeros/len(merged):.2%}), Ground {ground_zeros}/{len(merged)} ({ground_zeros/len(merged):.2%})")
            
            # Rename columns
            merged = merged.rename(columns={"Value_sat": "Satellite", "Value_ground": "Ground"})
            self.log(f"Renamed columns: {merged.columns.tolist()}")
            
            # Assign merged data to self.data
            self.data = merged
            self.log(f"Analysis using {self.agg_level_var.get()} {self.agg_func_var.get()} aggregation: {len(self.data)} rows")

            # Set storm threshold
            if self.threshold_mode_var.get() == "Percentile":
                storm_threshold = self.compute_75th_percentile_threshold(self.data['Ground'].values, self.agg_level_var.get())
                self.log(f"Storm threshold: {storm_threshold:.2f} mm (75th percentile, {self.agg_level_var.get()} aggregation)")
            else:
                storm_threshold = float(self.storm_entry.get().strip())
                self.log(f"Storm threshold: {storm_threshold:.2f} mm (Manual)")
            
            # Set other parameters
            quartile_points = self.get_quartile_points()
            clean_quartile_points = [int(q) for q in quartile_points]
            self.quartile_points = clean_quartile_points
            bias_correction_scope = self.bias_correction_scope_var.get()
            self.log(f"Using quartile points: {clean_quartile_points}")
            self.log(f"Storm threshold: {storm_threshold:.2f} mm")
            self.log(f"Zero-rain adjustment: {'Enabled' if self.zero_rain_var.get() else 'Disabled'}")
            self.log(f"Outlier capping: {'Enabled' if self.outlier_capping_var.get() else 'Disabled'}")
            self.log(f"Bias correction data scope: {bias_correction_scope}")

            # Check for sufficient non-zero data
            sat_nonzero = self.data['Satellite'][self.data['Satellite'] > 0]
            ground_nonzero = self.data['Ground'][self.data['Ground'] > 0]
            min_nonzero = 5 if self.agg_level_var.get() in ["Yearly", "Monthly"] else 20
            if len(sat_nonzero) < min_nonzero or len(ground_nonzero) < min_nonzero:
                self.log(f"Error: Insufficient non-zero data for {self.agg_level_var.get()} aggregation - Satellite: {len(sat_nonzero)}, Ground: {len(ground_nonzero)}")
                messagebox.showerror("Error", f"Insufficient non-zero data for {self.agg_level_var.get()} aggregation - Satellite: {len(sat_nonzero)}, Ground: {len(ground_nonzero)}")
                return

            if bias_correction_scope == "Full Satellite":
                sat_data = self.sat_df.set_index('Date')['Satellite'].dropna()
                self.log(f"Using full satellite data for bias correction: {len(sat_data)} days")
                ground_zero_frac = (self.data['Ground'] == 0).mean()
                if self.zero_rain_var.get():
                    sat_data = self.adjust_zero_rain(sat_data.values, self.data['Ground'].values, ground_zero_frac)
                    new_sat_zero_frac = (sat_data == 0).mean()
                    self.log(f"Preprocess (full satellite): Adjusted satellite dry days to {new_sat_zero_frac:.2%} (target {ground_zero_frac:.2%})")
                sat_small = (sat_data < MIN_RAINFALL_THRESHOLD) & (sat_data > 0)
                sat_data[sat_small] = 0
                self.log(f"Preprocess (full satellite): Satellite values < {MIN_RAINFALL_THRESHOLD} mm set to 0: {sat_small.sum()} points")
                if self.outlier_capping_var.get():
                    sat_nonzero = sat_data[sat_data > 0]
                    if len(sat_nonzero) > 0:
                        sat_max = np.percentile(sat_nonzero, 98)
                        sat_outliers = sat_data > sat_max
                        sat_data[sat_outliers] = sat_max
                        self.log(f"Preprocess (full satellite): Capped {sat_outliers.sum()} satellite outliers at {sat_max:.2f} mm")
            else:
                sat_data = self.data.set_index('Date')['Satellite']
                self.log(f"Using common dates for bias correction: {len(sat_data)} days")

            ground_data = self.data.set_index('Date')['Ground']
            common_dates = ground_data.index

            sat_zeros = (self.data['Satellite'] == 0).sum()
            ground_zeros = (self.data['Ground'] == 0).sum()
            total = len(self.data)
            self.log(f"Zero-rain days (common dates): Satellite {sat_zeros}/{total} ({sat_zeros/total:.2%}), Ground {ground_zeros}/{total} ({ground_zeros/total:.2%})")

            ground_storms = self.data[self.data['Ground'] > storm_threshold][['Date', 'Ground']].round(2)
            sat_storms = self.data[self.data['Satellite'] > storm_threshold][['Date', 'Satellite']].round(2)
            mismatches, offset_hist, weighted_median = self.analyze_storm_offsets(ground_storms, sat_storms)
            clean_offset_hist = [(int(o[0]), int(o[1])) for o in offset_hist]
            self.log(f"Preprocess: Storm offset distribution: {clean_offset_hist}")
            self.log(f"Preprocess: Weighted median offset: {int(weighted_median)} days")
            if weighted_median != 0 and abs(weighted_median) <= 3:
                self.data['Satellite'] = self.data['Satellite'].shift(weighted_median)
                self.data['Satellite'].fillna(0, inplace=True)
                sat_data = sat_data.shift(weighted_median).fillna(0)
                self.log(f"Preprocess: Shifted satellite data by {int(weighted_median)} days to align storms")
                sat_storms = self.data[self.data['Satellite'] > storm_threshold][['Date', 'Satellite']].round(2)
            self.log(f"Ground storms: {len(ground_storms)}, Satellite storms: {len(sat_storms)}, Mismatches: {mismatches}")

            ground_quartiles = np.percentile(ground_data, clean_quartile_points)
            satellite_quartiles = np.percentile(sat_data, clean_quartile_points)
            self.log(f"Preparing linear_quantile_mapping: sat_data shape={self.data['Satellite'].values.shape}, sat_quartiles={len(satellite_quartiles)}, ground_quartiles={len(ground_quartiles)}, quartile_points={len(clean_quartile_points)}")
            self.data['Satellite_Linear'] = self.linear_quantile_mapping(
                self.data['Satellite'].values,
                satellite_quartiles,
                ground_quartiles,
                clean_quartile_points
            )
            self.data['Satellite_Rank'] = self.rank_based_mapping(sat_data.loc[common_dates].values, ground_data.values)
            self.data['Satellite_Spline'] = self.spline_quantile_mapping(sat_data.loc[common_dates].values, ground_data.values)

            eval_log = self.evaluate_correction_methods(
                self.data['Ground'],
                self.data['Satellite_Linear'],
                self.data['Satellite_Rank'],
                self.data['Satellite_Spline']
            )
            for line in eval_log:
                self.log(line)

            rain_mask = self.predict_rain(self.data['Satellite'], self.data['Ground'])
            ground_dry = self.data['Ground'] == 0
            sat_dry_pred = ~rain_mask
            dry_agreement = np.mean(ground_dry == sat_dry_pred)
            self.log(f"Preprocess: Dry day prediction agreement: {dry_agreement:.2%}")

            spline_small = (self.data['Satellite_Spline'] < MIN_RAINFALL_THRESHOLD) & (self.data['Satellite_Spline'] > 0)
            self.log(f"Spline values < {MIN_RAINFALL_THRESHOLD} mm set to 0: {spline_small.sum()} points")

            self.metrics['Original'], log_text = self.evaluate_metrics(self.calculate_metrics(self.data['Ground'], self.data['Satellite'], storm_threshold), "Original")
            self.metrics['Linear'], log_text_linear = self.evaluate_metrics(self.calculate_metrics(self.data['Ground'], self.data['Satellite_Linear'], storm_threshold), "Linear")
            self.metrics['Rank'], log_text_rank = self.evaluate_metrics(self.calculate_metrics(self.data['Ground'], self.data['Satellite_Rank'], storm_threshold), "Rank")
            self.metrics['Spline'], log_text_spline = self.evaluate_metrics(self.calculate_metrics(self.data['Ground'], self.data['Satellite_Spline'], storm_threshold), "Spline")
            self.log(log_text + "\n" + log_text_linear + "\n" + log_text_rank + "\n" + log_text_spline)

            methods = ['Original', 'Linear', 'Rank', 'Spline']
            self.log("\nSummary of All Correction Methods:")
            for method in methods:
                metrics = self.metrics[method]
                summary_parts = []
                for metric in ['RMSE', 'Bias', 'FAR', 'KS']:
                    value = metrics.get(metric, np.nan)
                    crit = {
                        'RMSE': {'threshold': 10.0, 'unit': 'mm', 'good': '≤10.0', 'moderate': '≤20.0', 'poor': '>20.0', 'ref': 'Ebert et al. (2007)'},
                        'Bias': {'threshold': 1.0, 'unit': 'mm', 'good': '≤1.0', 'moderate': '≤3.0', 'poor': '>3.0', 'ref': 'Maggioni et al. (2016)'},
                        'FAR': {'threshold': 0.3, 'unit': '', 'good': '≤0.3', 'moderate': '≤0.6', 'poor': '>0.6', 'ref': 'Maggioni et al. (2016)'},
                        'KS': {'threshold': 0.05, 'unit': '', 'good': '≤0.05', 'moderate': '≤0.1', 'poor': '>0.1', 'ref': 'Wilks (2006)'}
                    }[metric]
                    if np.isnan(value):
                        status = 'NaN'
                    else:
                        if metric == 'Bias':
                            status = 'Good' if abs(value) <= crit['threshold'] else 'Moderate' if abs(value) <= float(crit['moderate'].split('≤')[1]) else 'Poor'
                        elif metric == 'KS':
                            status = 'Good' if value <= crit['threshold'] else 'Moderate' if value <= float(crit['moderate'].split('≤')[1]) else 'Poor'
                        else:
                            status = 'Good' if value <= crit['threshold'] else 'Moderate' if value <= float(crit['moderate'].split('≤')[1]) else 'Poor'
                        summary_parts.append(f"{metric} ({value:.3f} {crit['unit']}): {status}")
                self.log(f"{method}: {', '.join(summary_parts)}")

            self.generate_plots(save=True)
            self.update_metrics_table()
            self.display_plot(None)

            output_file = os.path.join(self.output_dir_var.get(), f"Rainfall_Comparison_With_Dates_{timestamp}_{self.agg_level_var.get().lower()}.xlsx")
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                self.data.to_excel(writer, sheet_name=f'Data_{self.agg_level_var.get()}', index=False)
                pd.DataFrame(self.metrics).to_excel(writer, sheet_name='Metrics')
                pd.DataFrame({
                    'Parameter': [
                        'Storm Threshold', 'Quartile Points', 'Zero-Rainfall', 'Outlier Capping',
                        'Spline Method', 'Bias Correction Scope', 'Time Series Start Date', 'Time Series End Date',
                        'Aggregation Level', 'Aggregation Function'
                    ],
                    'Value': [
                        storm_threshold, str(clean_quartile_points), str(self.zero_rain_var.get()),
                        str(self.outlier_capping_var.get()), self.spline_method_var.get(),
                        bias_correction_scope, self.start_date_var.get(), self.end_date_var.get(),
                        self.agg_level_var.get(), self.agg_func_var.get()
                    ]
                }).to_excel(writer, sheet_name='Parameters')
            self.log(f"Analysis complete. Results saved to {output_file}")

            sys.stdout = sys.__stdout__
            diagnostics = output_capture.getvalue()
            if diagnostics:
                self.log("Diagnostics:\n" + diagnostics)
            
            self.root.update()
        except Exception as e:
            sys.stdout = sys.__stdout__
            self.log(f"Error in run_analysis: {str(e)}")
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")

    def update_metrics_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree["columns"] = ("Metric", "Original", "Linear", "Rank", "Spline")
        self.tree.heading("Metric", text="Metric")
        self.tree.heading("Original", text="Original")
        self.tree.heading("Linear", text="Linear")
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("Spline", text="Spline")
        metrics_list = ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']
        for metric in metrics_list:
            values = [metric]
            for method in ['Original', 'Linear', 'Rank', 'Spline']:
                value = self.metrics.get(method, {}).get(metric, np.nan)
               
                values.append(f"{value:.3f}" if not np.isnan(value) else "NaN")
            self.tree.insert("", "end", values=values)

    def ecdf(self, data):
        sorted_data = np.sort(data)
        n = len(data)
        return sorted_data, np.arange(1, n + 1) / n

    def generate_plots(self, start_date=None, end_date=None, timestamp=None, save=True):
        # Ensure output directory exists if saving
        if save:
            os.makedirs(self.output_dir_var.get(), exist_ok=True)
            timestamp = timestamp or pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            self.log(f"Generating plots with timestamp: {timestamp}")

        # Clear existing plots
        self.plots.clear()

        # Histogram Plot
        fig_hist = plt.figure(figsize=(12, 6))
        bins = int(np.sqrt(len(self.data['Satellite'])))
        bin_edges = np.histogram_bin_edges(self.data['Satellite'].dropna(), bins=bins)
        mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.hist(self.data['Ground'].dropna(), bins=bin_edges, alpha=0.2, label='Ground', color='blue')
        counts, _ = np.histogram(self.data['Ground'].dropna(), bins=bin_edges)
        plt.plot(mids, counts, 'k-', label='Ground (line)', color='blue', linewidth=2)
        plt.hist(self.data['Satellite'].dropna(), bins=bin_edges, alpha=0.2, label='Satellite (Original)', color='red')
        counts, _ = np.histogram(self.data['Satellite'].dropna(), bins=bin_edges)
        plt.plot(mids, counts, 'r--', label='Satellite (Original line)', color='red', linewidth=2)
        plt.hist(self.data['Satellite_Linear'].dropna(), bins=bin_edges, alpha=0.2, label='Satellite (Linear)', color='green')
        counts, _ = np.histogram(self.data['Satellite_Linear'].dropna(), bins=bin_edges)
        plt.plot(mids, counts, 'g-', label='Satellite (Linear line)', color='green', linewidth=2)
        plt.hist(self.data['Satellite_Spline'].dropna(), bins=bin_edges, alpha=0.2, label=f'Satellite (Spline {self.spline_method_var.get()})', color='orange')
        counts, _ = np.histogram(self.data['Satellite_Spline'].dropna(), bins=bin_edges)
        plt.plot(mids, counts, 'y-', label=f'Satellite (Spline {self.spline_method_var.get()} line)', color='orange', linewidth=3)
        plt.hist(self.data['Satellite_Rank'].dropna(), bins=bin_edges, alpha=0.2, label='Satellite (Rank)', color='#800080')
        counts, _ = np.histogram(self.data['Satellite_Rank'].dropna(), bins=bin_edges)
        plt.plot(mids, counts, linestyle='--', label='Satellite (Rank line)', color='#800080', linewidth=3)
        plt.legend()
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Density')
        plt.title(f'Rainfall: Distributions ({bins} bins)')
        self.plots['Histogram'] = fig_hist
        if save:
            plt.savefig(os.path.join(self.output_dir_var.get(), f'histogram_comparison_{timestamp}.png'))
            self.log("Generated and saved Histogram plot")
        plt.close(fig_hist)

        # CDF Plot
        fig_cdf = plt.figure(figsize=(12, 6))
        x_g, y_g = self.ecdf(self.data['Ground'].dropna())

        x_s, y_s = self.ecdf(self.data['Satellite'].dropna())
        x_l, y_l = self.ecdf(self.data['Satellite_Linear'].dropna())
        x_r, y_r = self.ecdf(self.data['Satellite_Rank'].dropna())
       
        x_sp, y_sp = self.ecdf(self.data['Satellite_Spline'].dropna())
        plt.plot(x_g, y_g, label='Ground', color='blue', linewidth=2)
        plt.plot(x_s, y_s, label='Satellite (Original)', color='red', linewidth=2)
        plt.plot(x_l, y_l, label='Satellite (Linear)', color='green', linewidth=2)
        plt.plot(x_r, y_r, label='Satellite (Rank)', color='#800080', alpha=0.7, linewidth=3)
        plt.plot(x_sp, y_sp, label=f'Satellite (Spline {self.spline_method_var.get()})', color='orange', linestyle='-.', linewidth=3)
        plt.legend()
        plt.xlabel('Rainfall (mm)')
        plt.ylabel('Cumulative Probability')
        plt.title('Empirical CDFs')
        plt.grid(True)
        self.plots['CDF'] = fig_cdf
        if save:
            plt.savefig(os.path.join(self.output_dir_var.get(), f'cdf_comparison_{timestamp}.png'))
            self.log("Generated and saved CDF plot")
        plt.close(fig_cdf)

        # Time Series Plot
        fig_ts = plt.figure(figsize=(12, 6))
        if start_date and end_date:
            valid_data = self.data[(self.data['Date'] >= start_date) & (self.data['Date'] <= end_date)]
            title = f'Time Series ({start_date.date()} to {end_date.date()})'
        else:
            valid_data = self.data.head(100)
            title = 'Time Series (First 100 Days)'
        plt.plot(valid_data['Date'], valid_data['Ground'], label='Ground', color='blue', alpha=0.5, linewidth=2)
        plt.plot(valid_data['Date'], valid_data['Satellite'], label='Satellite (Original)', color='red', alpha=0.5, linewidth=2)
        plt.plot(valid_data['Date'], valid_data['Satellite_Linear'], label='Satellite (Linear)', color='green', alpha=0.5, linewidth=2)
        plt.plot(valid_data['Date'], valid_data['Satellite_Rank'], label='Satellite (Rank)', color='#800080', alpha=0.7, linewidth=3)
        plt.plot(valid_data['Date'], valid_data['Satellite_Spline'], label=f'Satellite (Spline {self.spline_method_var.get()})', color='orange', alpha=0.7, linewidth=3)
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Rainfall (mm)')
        plt.title(title)
        plt.xticks(rotation=45)
        plt.tight_layout()
        self.plots['Time Series'] = fig_ts
        if save:
            plt.savefig(os.path.join(self.output_dir_var.get(), f'timeseries_comparison_{timestamp}.png'))
            self.log("Generated and saved Time Series plot")
        plt.close(fig_ts)

        # Metrics Comparison Plot
        fig_metrics = plt.figure(figsize=(12, 6))
        metrics_data = {
            'Method': ['Original', 'Linear', 'Rank', 'Spline'],
            'RMSE': [self.metrics.get('Original', {}).get('RMSE', np.nan), self.metrics.get('Linear', {}).get('RMSE', np.nan),
                    self.metrics.get('Rank', {}).get('RMSE', np.nan), self.metrics.get('Spline', {}).get('RMSE', np.nan)],
            'MAE': [self.metrics.get('Original', {}).get('MAE', np.nan), self.metrics.get('Linear', {}).get('MAE', np.nan),
                    self.metrics.get('Rank', {}).get('MAE', np.nan), self.metrics.get('Spline', {}).get('MAE', np.nan)],
            'POD': [self.metrics.get('Original', {}).get('POD', np.nan), self.metrics.get('Linear', {}).get('POD', np.nan),
                    self.metrics.get('Rank', {}).get('POD', np.nan), self.metrics.get('Spline', {}).get('POD', np.nan)],
            'KGE': [self.metrics.get('Original', {}).get('KGE', np.nan), self.metrics.get('Linear', {}).get('KGE', np.nan),
                    self.metrics.get('Rank', {}).get('KGE', np.nan), self.metrics.get('Spline', {}).get('KGE', np.nan)],
            'KS': [self.metrics.get('Original', {}).get('KS', np.nan), self.metrics.get('Linear', {}).get('KS', np.nan),
                self.metrics.get('Rank', {}).get('KS', np.nan), self.metrics.get('Spline', {}).get('KS', np.nan)]
        }
        self.log(f"Metrics data for plotting: {metrics_data}")
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.plot(x='Method', kind='bar', ax=fig_metrics.gca())
        plt.ylabel('Metric Value')
        plt.title('Performance Metrics Comparison')
        plt.legend(loc='best')
        plt.tight_layout()
        self.plots['Metrics Comparison'] = fig_metrics
        if save:
            save_path = os.path.join(self.output_dir_var.get(), f'metrics_comparison_{timestamp}.png')
            plt.savefig(save_path)
            self.log(f"Generated and saved Metrics Comparison plot to {save_path}")
        plt.close(fig_metrics)

        self.log(f"Generated plots: {list(self.plots.keys())}")

    def display_plot(self, event):
        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            plot_name = self.plot_var.get()
            self.log(f"Displaying plot: {plot_name}, exists in self.plots: {plot_name in self.plots}")
            if plot_name in self.plots:
                fig = self.plots[plot_name]
                self.log(f"Plot {plot_name} figure type: {type(fig).__name__}")
                canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill="both", expand=True)
                self.log(f"Successfully displayed {plot_name}")
            else:
                self.log(f"Error: {plot_name} not found in self.plots")
                messagebox.showerror("Error", f"Plot {plot_name} not available. Please run analysis first.")
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
                'Metric': ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS'],
                'Original': [self.metrics.get('Original', {}).get(m, np.nan) for m in ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']],
                'Linear': [self.metrics.get('Linear', {}).get(m, np.nan) for m in ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']],
                'Rank': [self.metrics.get('Rank', {}).get(m, np.nan) for m in ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']],
                'Spline': [self.metrics.get('Spline', {}).get(m, np.nan) for m in ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']]
            }
            pd.DataFrame(metrics_data).to_csv(file_path, index=False)
            self.log(f"Metrics saved to {file_path}")

    def save_plot(self):
        try:
            plot_name = self.plot_var.get()
            if plot_name in self.plots or plot_name == 'Metrics Comparison':
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                suggested_filename = f"{plot_name.lower().replace(' ', '_')}_{timestamp}.png"
                file_path = filedialog.asksaveasfilename(
                    initialdir=self.output_dir_var.get(),
                    initialfile=suggested_filename,
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png")]
                )
                if file_path:
                    start_date = pd.to_datetime(self.start_date_var.get()) if self.start_date_var.get() else None
                    end_date = pd.to_datetime(self.end_date_var.get()) if self.end_date_var.get() else None
                    self.generate_plots(start_date=start_date, end_date=end_date, timestamp=timestamp, save=True)
                    self.log(f"Plot {plot_name} saved to {file_path}")
                    self.display_plot(None)
            else:
                self.log(f"Error: {plot_name} not available for saving")
                messagebox.showerror("Error", f"Plot {plot_name} not available. Please run analysis first.")
        except Exception as e:
            self.log(f"Error saving plot {plot_name}: {e}")
            messagebox.showerror("Error", f"Failed to save {plot_name}: {e}")

    import numpy as np

    def linear_quantile_mapping(self, sat_data, sat_quantiles, ground_quantiles, quartile_points):
        # Input validation for quantiles
        finite_quantile_mask = np.isfinite(sat_quantiles) & np.isfinite(ground_quantiles)
        if not np.all(finite_quantile_mask):
            self.log(f"Linear: Dropped {np.sum(~finite_quantile_mask)} non-finite quantile points")
            sat_quantiles = sat_quantiles[finite_quantile_mask]
            ground_quantiles = ground_quantiles[finite_quantile_mask]
            quartile_points = quartile_points[finite_quantile_mask]
        
        if len(quartile_points) < 2:
            self.log("Linear: Error - Fewer than 2 valid quantile points after filtering; returning empty array")
            return np.array([], dtype=float)

        # Input validation for sat_data
        finite_data_mask = np.isfinite(sat_data)
        if not np.all(finite_data_mask):
            self.log(f"Linear: Dropped {np.sum(~finite_data_mask)} rows with non-finite values in sat_data")
            sat_data = sat_data[finite_data_mask]
        
        if len(sat_data) == 0:
            self.log("Linear: Error - No finite data remaining after filtering; returning empty array")
            return np.array([], dtype=float)

        adjusted = np.zeros_like(sat_data, dtype=float)
        n_points = len(quartile_points)

        # Lower bound (sat_data <= first quantile)
        mask_lower = sat_data <= sat_quantiles[0]
        if sat_quantiles[0] > 0:
            scale = ground_quantiles[0] / sat_quantiles[0]
            adjusted[mask_lower] = np.clip(sat_data[mask_lower] * scale, 0, ground_quantiles[0])
        else:
            adjusted[mask_lower] = ground_quantiles[0]
        self.log(f"Linear: Applied lower bound mapping for {mask_lower.sum()} points")

        # Interpolate between quantile points
        for i in range(n_points - 1):
            mask = (sat_data > sat_quantiles[i]) & (sat_data <= sat_quantiles[i + 1])
            denominator = sat_quantiles[i + 1] - sat_quantiles[i]
            if denominator > 1e-6:  # Safe threshold to avoid division issues
                slope = (ground_quantiles[i + 1] - ground_quantiles[i]) / denominator
                adjusted[mask] = ground_quantiles[i] + (sat_data[mask] - sat_quantiles[i]) * slope
            else:
                adjusted[mask] = ground_quantiles[i]
            if mask.sum() > 0:
                self.log(f"Linear: Interpolated {mask.sum()} points between quantiles {quartile_points[i]} and {quartile_points[i+1]}")

        # Upper bound (sat_data > last quantile)
        mask_upper = sat_data > sat_quantiles[-1]
        if sat_quantiles[-1] > 1e-6:
            scale = ground_quantiles[-1] / sat_quantiles[-1]
            adjusted[mask_upper] = ground_quantiles[-1] + (sat_data[mask_upper] - sat_quantiles[-1]) * scale
            adjusted[mask_upper] = np.clip(adjusted[mask_upper], ground_quantiles[-1], np.max(ground_quantiles) * 1.5)
        else:
            adjusted[mask_upper] = ground_quantiles[-1]
        self.log(f"Linear: Applied upper bound mapping for {mask_upper.sum()} points")

        # Ensure non-negative and finite results
        adjusted = np.clip(adjusted, 0, None)
        if not np.all(np.isfinite(adjusted)):
            self.log("Linear: Warning - Non-finite values in adjusted output; dropping them")
            finite_adj_mask = np.isfinite(adjusted)
            adjusted = adjusted[finite_adj_mask]
            self.log(f"Linear: Dropped {np.sum(~finite_adj_mask)} non-finite values from adjusted output")

        self.log(f"Linear: Final adjusted stats - mean={adjusted.mean():.2f} mm, zeros={(adjusted == 0).mean():.2%}, length={len(adjusted)}")
        return adjusted

    def calculate_metrics(self, observed, predicted, storm_threshold=12.0):
        # Ensure inputs are pd.Series with date indices
        if not isinstance(observed, pd.Series) or not isinstance(predicted, pd.Series):
            self.log("Inputs must be pd.Series with date indices")
            return {}
        
        # Align data by common dates
        observed = observed.dropna()
        predicted = predicted.dropna()
        common_idx = observed.index.intersection(predicted.index)
        if len(common_idx) == 0:
            self.log("No common dates found for metrics calculation")
            return {}
        
        observed = observed.loc[common_idx]
        predicted = predicted.loc[common_idx]
        
        # Log alignment details
        self.log(f"Aligned data: {len(common_idx)} common dates")
        self.log(f"Ground stats: mean={np.mean(observed):.2f} mm, std={np.std(observed):.2f} mm, zeros={(observed == 0).sum()/len(observed)*100:.2f}%")
        self.log(f"Satellite stats: mean={np.mean(predicted):.2f} mm, std={np.std(predicted):.2f} mm, zeros={(predicted == 0).sum()/len(predicted)*100:.2f}%")
        
        # Calculate other metrics
        mse = mean_squared_error(observed, predicted)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(observed, predicted)
        bias = np.mean(predicted - observed)
        
        # Categorical metrics for storm events
        rain_events_obs = observed > storm_threshold
        rain_events_pred = predicted > storm_threshold
        hits = np.sum(rain_events_obs & rain_events_pred)
        misses = np.sum(rain_events_obs & ~rain_events_pred)
        false_alarms = np.sum(~rain_events_obs & rain_events_pred)
        pod = hits / (hits + misses) if (hits + misses) > 0 else np.nan
        far = false_alarms / (hits + false_alarms) if (hits + false_alarms) > 0 else np.nan
        
        # Calculate KGE
        kge_components = self.calculate_kge(observed, predicted)
        kge = kge_components.get('KGE', np.nan)
        
        # Calculate KS statistic
        ks_stat, _ = ks_2samp(observed, predicted) if len(observed) > 0 and len(predicted) > 0 else (np.nan, np.nan)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'Bias': bias,
            'POD': pod,
            'FAR': far,
            'KGE': kge,
            'KS': ks_stat
        }
    
    def calculate_kge(self, observed, predicted):
        try:
            if len(observed) == 0 or len(predicted) == 0:
                return {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
            r = np.corrcoef(observed, predicted)[0, 1]
            if np.isnan(r):
                return {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}
            alpha = np.std(predicted) / np.std(observed) if np.std(observed) != 0 else np.nan
            beta = np.mean(predicted) / np.mean(observed) if np.mean(observed) != 0 else np.nan
            kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
            return {'KGE': kge, 'r': r, 'alpha': alpha, 'beta': beta}
        except Exception as e:
            self.log(f"Error calculating KGE: {e}")
            return {'KGE': np.nan, 'r': np.nan, 'alpha': np.nan, 'beta': np.nan}

    def evaluate_metrics(self, metrics, method):
        criteria = {
            'RMSE': {'threshold': 10.0, 'unit': 'mm', 'good': '≤10.0', 'moderate': '≤20.0', 'poor': '>20.0', 'ref': 'Ebert et al. (2007)'},
            'MAE': {'threshold': 5.0, 'unit': 'mm', 'good': '≤5.0', 'moderate': '≤10.0', 'poor': '>10.0', 'ref': 'Tian et al. (2010)'},
            'Bias': {'threshold': 1.0, 'unit': 'mm', 'good': '≤1.0', 'moderate': '≤3.0', 'poor': '>3.0', 'ref': 'Maggioni et al. (2016)'},
            'POD': {'threshold': 0.7, 'unit': '', 'good': '≥0.7', 'moderate': '≥0.3', 'poor': '<0.3', 'ref': 'Ebert et al. (2007)'},
            'FAR': {'threshold': 0.3, 'unit': '', 'good': '≤0.3', 'moderate': '≤0.6', 'poor': '>0.6', 'ref': 'Maggioni et al. (2016)'},
            'KGE': {'threshold': 0.5, 'unit': '', 'good': '≥0.5', 'moderate': '≥0', 'poor': '<0', 'ref': 'Kling et al. (2012)'},
            'KS': {'threshold': 0.05, 'unit': '', 'good': '≤0.05', 'moderate': '≤0.1', 'poor': '>0.1', 'ref': 'Wilks (2006)'}
        }
        log_lines = [f"\n{method} Performance Metrics:"]
        for metric, value in metrics.items():
            if np.isnan(value):
                log_lines.append(f"{metric}: NaN")
                continue
            crit = criteria.get(metric, None)
            if crit is None:
                log_lines.append(f"{metric}: {value:.3f} (No criteria defined)")
                continue
            if metric in ['POD', 'KGE']:
                status = 'Good' if value >= crit['threshold'] else 'Moderate' if value >= float(crit['moderate'].replace('≥', '')) else 'Poor'
            elif metric == 'Bias':
                status = 'Good' if abs(value) <= crit['threshold'] else 'Moderate' if abs(value) <= float(crit['moderate'].split('≤')[1]) else 'Poor'
            else:
                status = 'Good' if value <= crit['threshold'] else 'Moderate' if value <= float(crit['moderate'].split('≤')[1]) else 'Poor'
            log_lines.append(f"{metric}: {value:.3f} {crit['unit']} ({status}, {crit['ref']})")
        return metrics, "\n".join(log_lines)

    def analyze_storm_offsets(self, ground_storms, sat_storms, max_lag=5):
        mismatches = 0
        offsets = []
        weights = []
        for g_date, g_rain in ground_storms.itertuples(index=False):
            g_date = pd.Timestamp(g_date)
            nearby_sat = sat_storms[(sat_storms['Date'] >= g_date - pd.Timedelta(days=max_lag)) &
                                    (sat_storms['Date'] <= g_date + pd.Timedelta(days=max_lag))]
            if nearby_sat.empty:
                mismatches += 1
            else:
                closest_sat = nearby_sat.iloc[np.argmin(np.abs(nearby_sat['Date'] - g_date))]
                offset = (closest_sat['Date'] - g_date).days
                offsets.append(offset)
                weights.append(g_rain)
        weighted_median = 0
        offset_hist = []
        if offsets:
            bins = list(range(-max_lag, max_lag + 2))
            hist, _ = np.histogram(offsets, bins=bins)
            offset_hist = list(zip(bins[:-1], hist))
            print(f"Preprocess: Storm offset distribution: {offset_hist}")
            if weights:
                sorted_pairs = sorted(zip(offsets, weights), key=lambda x: x[0])
                sorted_offsets, sorted_weights = zip(*sorted_pairs)
                cum_weights = np.cumsum(sorted_weights)
                total_weight = cum_weights[-1]
                median_idx = np.searchsorted(cum_weights, total_weight / 2)
                weighted_median = sorted_offsets[median_idx]
                print(f"Preprocess: Weighted median offset: {weighted_median} days")
        return mismatches, offset_hist, weighted_median

if __name__ == "__main__":
    root = ttk.Window()
    app = RainfallAnalysisGUI(root)
    root.mainloop()
