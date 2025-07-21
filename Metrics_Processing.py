import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
from datetime import datetime
import os
import glob
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import logging

# Centralized column names
DATE_COLUMN = 'Date'
MONTH_COLUMN = 'Date'  # Monthly data uses 'Date' for YYYY-MM
YEAR_COLUMN = 'Year'
SATELLITE_COLUMN = 'Satellite'
GROUND_COLUMN = 'Ground'
METHOD_COLUMNS = ['Satellite_Linear', 'Satellite_Rank', 'Satellite_Spline']
METHODS = ['Original', 'Linear', 'Rank', 'Spline']
METRICS = ['RMSE', 'MAE', 'Bias', 'POD', 'FAR', 'KGE', 'KS']

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Metrics to normalize
metrics_to_normalize = ['RMSE', 'MAE', 'Bias']

class MetricsProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Rainfall Metrics Processor")
        self.style = Style(theme='darkly')
        
        # Variables
        default_dir = os.path.abspath(os.getcwd())
        self.input_dir = tk.StringVar(value=default_dir)
        self.output_file = tk.StringVar()
        self.aggregation_type = tk.StringVar(value='Daily')
        self.file_list = []
        
        # GUI Layout
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        # Make main_frame expandable
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(2, weight=1)  # Row for tree_frame
        self.main_frame.grid_columnconfigure(1, weight=1)  # Column for widgets
        
        # Input Directory
        ttk.Label(self.main_frame, text="Input Directory:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.input_dir, width=50).grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(self.main_frame, text="Browse", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Aggregation Type
        ttk.Label(self.main_frame, text="Aggregation Type:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(self.main_frame, textvariable=self.aggregation_type, 
                     values=['Daily', 'Monthly Mean', 'Yearly max'], state='readonly').grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # File List Display
        self.tree_frame = ttk.Frame(self.main_frame)
        self.tree_frame.grid(row=2, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tree_frame.grid_rowconfigure(0, weight=1)
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree = ttk.Treeview(self.tree_frame, columns=('File',), show='headings', height=10)
        self.tree.heading('File', text='Selected Files')
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.tree_scrollbar = ttk.Scrollbar(self.tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=self.tree_scrollbar.set)
        
        # Output File
        ttk.Label(self.main_frame, text="Output File:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(self.main_frame, textvariable=self.output_file, width=50).grid(row=3, column=1, padx=5, pady=5, sticky=(tk.W, tk.E))
        ttk.Button(self.main_frame, text="Browse", command=self.browse_output_file).grid(row=3, column=2, padx=5, pady=5)
        
        # Calculate Metrics Button
        self.calculate_button = ttk.Button(self.main_frame, text="Calculate Metrics", style=PRIMARY, command=self.process_files)
        self.calculate_button.grid(row=4, column=0, columnspan=3, pady=10, sticky=tk.EW)
        logging.info("Calculate Metrics button created and placed at row=4, column=0, columnspan=3")
        
        # Log Display
        self.log_text = tk.Text(self.main_frame, height=5, width=60)
        self.log_text.grid(row=5, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E))
        self.log_scrollbar = ttk.Scrollbar(self.main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_scrollbar.grid(row=5, column=3, sticky=(tk.N, tk.S))
        self.log_text.configure(yscrollcommand=self.log_scrollbar.set)
        
        # Initialize file list
        self.update_file_list()
        
    def log_message(self, message):
        try:
            self.log_text.insert(tk.END, f"{message}\n")
            self.log_text.yview_moveto(1.0)
            logging.info(message)
        except Exception as e:
            logging.error(f"Error updating log: {e}")
        
    def browse_input_dir(self):
        directory = filedialog.askdirectory(initialdir=self.input_dir.get())
        if directory:
            self.input_dir.set(os.path.abspath(directory))
            self.update_file_list()
            
    def browse_output_file(self):
        file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel files", "*.xlsx")],
                                           initialfile=f"Metrics_Summary_{self.aggregation_type.get()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                           initialdir=self.input_dir.get())
        if file:
            self.output_file.set(os.path.abspath(file))
            
    def update_file_list(self):
        self.tree.delete(*self.tree.get_children())
        self.file_list = []
        agg_type = self.aggregation_type.get()
        input_dir = self.input_dir.get()
        
        if not os.path.exists(input_dir):
            self.log_message(f"Input directory does not exist: {input_dir}")
            self.log_message(f"Falling back to current working directory: {os.getcwd()}")
            self.input_dir.set(os.getcwd())
            input_dir = os.getcwd()
        
        pattern = r'([A-Z]+\d+)_(CHIRPS|GPM)_(daily|monmean|ymax)\.xlsx$'
        try:
            for file in os.listdir(input_dir):
                if re.match(pattern, file, re.IGNORECASE):
                    if (agg_type == 'Daily' and 'daily' in file.lower()) or \
                       (agg_type == 'Monthly Mean' and 'monmean' in file.lower()) or \
                       (agg_type == 'Yearly max' and 'ymax' in file.lower()):
                        self.file_list.append(file)
                        self.tree.insert('', tk.END, values=(file,))
            self.log_message(f"Found {len(self.file_list)} files for {agg_type} processing in {input_dir}")
            if not self.file_list:
                self.log_message(f"No matching files found in {input_dir}. Expected files like A1_CHIRPS_monmean.xlsx, B1_GPM_monmean.xlsx, etc.")
        except Exception as e:
            self.log_message(f"Error accessing directory {input_dir}: {e}")
            messagebox.showerror("Error", f"Cannot access directory {input_dir}: {e}")
        
    def process_files(self):
        input_dir = self.input_dir.get()
        output_file = self.output_file.get()
        agg_type = self.aggregation_type.get()
        
        if not os.path.exists(input_dir):
            messagebox.showerror("Error", f"Input directory does not exist: {input_dir}")
            return
        if not output_file:
            messagebox.showerror("Error", "Please select an output file")
            return
        output_dir = os.path.dirname(output_file)
        if not os.path.exists(output_dir):
            messagebox.showerror("Error", f"Output directory does not exist: {output_dir}")
            return
            
        self.log_message(f"Starting metrics processing for {agg_type} data in {input_dir}...")
        all_metrics = []
        data_stats = []  # Store data means for Data_Means sheet
        locations = []
        storm_thresholds = {}  # Store storm thresholds with (location, dataset) keys
        data_means = {}  # Store means of data columns for each file for normalization
        
        # Process files
        file_pattern = os.path.join(input_dir, f"*_{'daily' if agg_type == 'Daily' else 'monmean' if agg_type == 'Monthly Mean' else 'ymax'}.xlsx")
        files = glob.glob(file_pattern, recursive=False)
        self.log_message(f"Found files: {files}")
        
        for file_path in files:
            filename = os.path.basename(file_path).lower()
            match = re.match(r'([A-Z]+\d+)_(CHIRPS|GPM)_(daily|monmean|ymax)\.xlsx$', filename, re.IGNORECASE)
            if not match:
                self.log_message(f"Skipping file {filename}: does not match pattern")
                continue
            location, dataset, file_type = match.groups()
            location = location.upper()  # Normalize to uppercase
            dataset = dataset.upper()
            if (agg_type == 'Daily' and file_type.lower() != 'daily') or \
               (agg_type == 'Monthly Mean' and file_type.lower() != 'monmean') or \
               (agg_type == 'Yearly Max' and file_type.lower() != 'ymax'):
                self.log_message(f"Skipping file {filename}: incorrect aggregation type")
                continue
                
            locations.append(location)
            self.log_message(f"Processing file: {file_path}, Location: {location}, Dataset: {dataset}")
            
            try:
                # Read Metrics sheet
                try:
                    df_metrics = pd.read_excel(file_path, sheet_name='Metrics', index_col=0)
                    self.log_message(f"Metrics sheet found in {file_path}. Columns: {df_metrics.columns.tolist()}, Index: {df_metrics.index.tolist()}")
                    expected_metrics = set(METRICS)
                    expected_methods = set(METHODS)
                    if not all(metric in df_metrics.index for metric in expected_metrics):
                        self.log_message(f"Error: Missing metrics in {file_path}. Expected: {expected_metrics}, Found: {set(df_metrics.index)}")
                        continue
                    if not all(method in df_metrics.columns for method in expected_methods):
                        self.log_message(f"Error: Missing methods in {file_path}. Expected: {expected_methods}, Found: {set(df_metrics.columns)}")
                        continue
                    df_metrics = df_metrics.apply(pd.to_numeric, errors='coerce')
                except Exception as e:
                    self.log_message(f"Error reading Metrics sheet in {file_path}: {e}")
                    messagebox.showerror("Error", f"Failed to read Metrics sheet in {file_path}: {e}")
                    continue
                
                # Read Parameters sheet for storm threshold
                try:
                    df_params = pd.read_excel(file_path, sheet_name='Parameters')
                    self.log_message(f"Parameters sheet found in {file_path}. DataFrame:\n{df_params.to_string()}")
                    if 'Parameter' in df_params.columns and 'Value' in df_params.columns:
                        storm_row = df_params[df_params['Parameter'] == 'Storm Threshold']
                        if not storm_row.empty:
                            storm_threshold = float(storm_row['Value'].iloc[0])
                            self.log_message(f"Storm Threshold for {location}_{dataset}: {storm_threshold}")
                        else:
                            storm_threshold = 0 if agg_type == 'Yearly max' else 0.00
                            self.log_message(f"'Storm Threshold' not found in Parameters sheet of {file_path}. Using default: {storm_threshold}")
                    else:
                        storm_threshold = 0 if agg_type == 'Yearly max' else 0.00
                        self.log_message(f"'Parameter' or 'Value' column not found in Parameters sheet of {file_path}. Using default Storm Threshold: {storm_threshold}")
                    storm_thresholds[(location, dataset)] = storm_threshold
                except Exception as e:
                    storm_threshold = 0 if agg_type == 'Yearly max' else 0.00
                    self.log_message(f"Error reading Parameters sheet in {file_path}: {e}. Using default Storm Threshold: {storm_threshold}")
                    storm_thresholds[(location, dataset)] = storm_threshold
                
                # Read Data sheet based on aggregation type
                data_sheet_name = 'Data_Daily' if agg_type == 'Daily' else 'Data_Monthly' if agg_type == 'Monthly Mean' else 'Data_Yearly'
                try:
                    df_data = pd.read_excel(file_path, sheet_name=data_sheet_name)
                    self.log_message(f"{data_sheet_name} sheet found in {file_path}. Columns: {df_data.columns.tolist()}")
                    # Compute means for relevant columns
                    data_columns = {
                        'Satellite': 'Original',
                        'Satellite_Linear': 'Linear',
                        'Satellite_Rank': 'Rank',
                        'Satellite_Spline': 'Spline'
                    }
                    file_data_stats = {'Location': location, 'Dataset': dataset}
                    file_data_means = {}
                    for col, method in data_columns.items():
                        if col in df_data.columns:
                            mean_value = df_data[col].mean()
                            file_data_stats[col] = mean_value
                            file_data_means[f"{location}_{dataset}_{method}"] = mean_value
                            self.log_message(f"Mean for {col} ({method}) in {file_path}: {mean_value}")
                        else:
                            self.log_message(f"Column {col} not found in {data_sheet_name} sheet of {file_path}")
                            file_data_stats[col] = np.nan
                            file_data_means[f"{location}_{dataset}_{method}"] = np.nan
                    data_stats.append(file_data_stats)
                    data_means.update(file_data_means)
                except Exception as e:
                    self.log_message(f"Error reading {data_sheet_name} sheet in {file_path}: {e}")
                    file_data_stats = {'Location': location, 'Dataset': dataset}
                    for col, method in data_columns.items():
                        file_data_stats[col] = np.nan
                        file_data_means[f"{location}_{dataset}_{method}"] = np.nan
                    data_stats.append(file_data_stats)
                    data_means.update(file_data_means)
                except Exception as e:
                    self.log_message(f"Error reading Data_Daily sheet in {file_path}: {e}")
                    file_data_stats = {'Location': location, 'Dataset': dataset}
                    for col, method in data_columns.items():
                        file_data_stats[col] = np.nan
                        file_data_means[f"{location}_{dataset}_{method}"] = np.nan
                    data_stats.append(file_data_stats)
                    data_means.update(file_data_means)
                
                # Extract metrics
                row = {'Location': location, 'Dataset': dataset}
                for metric in METRICS:
                    for method in METHODS:
                        value = df_metrics.loc[metric, method] if metric in df_metrics.index and method in df_metrics.columns else np.nan
                        row[f"{metric}_{method}"] = float(value) if pd.notna(value) else np.nan
                all_metrics.append(row)
            except Exception as e:
                self.log_message(f"Error processing {file_path}: {e}")
        
        if not all_metrics:
            messagebox.showerror("Error", f"No valid data processed from {input_dir}. Check file names or Metrics sheets.")
            return
            
        # Create Metrics DataFrame
        metrics_df = pd.DataFrame(all_metrics)
        if 'Location' not in metrics_df.columns:
            messagebox.showerror("Error", "Location column missing in DataFrame. Check file processing.")
            return
        metrics_df['Location'] = metrics_df['Location'].str.upper()
        metrics_df['Dataset'] = metrics_df['Dataset'].str.upper()
        metrics_df = metrics_df.set_index(['Location', 'Dataset'])
        self.log_message(f"Metrics DataFrame columns: {metrics_df.columns.tolist()}")
        self.log_message(f"Metrics DataFrame index: {metrics_df.index.tolist()}")
        
        # Create Data_Means DataFrame
        data_stats_df = pd.DataFrame(data_stats)
        if not data_stats_df.empty:
            data_stats_df = data_stats_df.set_index(['Location', 'Dataset'])
            self.log_message(f"Data_Means DataFrame columns: {data_stats_df.columns.tolist()}")
            self.log_message(f"Data_Means DataFrame index: {data_stats_df.index.tolist()}")
        else:
            self.log_message("No data means collected. Data_Means sheet will be empty.")
        
        # Create Storm_Thresholds DataFrame
        self.log_message(f"Storm Thresholds before creating DataFrame: {storm_thresholds}")
        storm_thresholds_df = pd.DataFrame.from_dict(
            {loc_dataset: threshold for loc_dataset, threshold in storm_thresholds.items()},
            orient='index', columns=['Storm_Threshold']
        )
        storm_thresholds_df.index = pd.MultiIndex.from_tuples(
            storm_thresholds_df.index, names=['Location', 'Dataset']
        )
        self.log_message(f"Storm_Thresholds DataFrame index: {storm_thresholds_df.index.tolist()}")
        self.log_message(f"Storm_Thresholds DataFrame values:\n{storm_thresholds_df.to_string()}")
        
        # Compute statistics
        stats = pd.DataFrame(index=['Mean', 'Std', 'Median', 'IQR', 'Data_Mean'], columns=metrics_df.columns)
        for col in metrics_df.columns:
            stats.loc['Mean', col] = metrics_df[col].mean()
            stats.loc['Std', col] = metrics_df[col].std()
            stats.loc['Median', col] = metrics_df[col].median()
            stats.loc['IQR', col] = metrics_df[col].quantile(0.75) - metrics_df[col].quantile(0.25)
            # Assign Data_Mean based on the corresponding method
            method = col.split('_')[-1]
            loc_dataset = metrics_df.index[metrics_df.index.get_loc(metrics_df.index[metrics_df[col].notna()][0])][0] + '_' + metrics_df.index[metrics_df.index.get_loc(metrics_df.index[metrics_df[col].notna()][0])][1]
            stats.loc['Data_Mean', col] = data_means.get(f"{loc_dataset}_{method}", np.nan)
        
        # Normalize metrics (only Mean_Metrics)
        mean_df = metrics_df.copy()
        for col in metrics_df.columns:
            data_mean = stats.loc['Data_Mean', col]
            method = col.split('_')[-1]
            loc_dataset = metrics_df.index[metrics_df.index.get_loc(metrics_df.index[metrics_df[col].notna()][0])][0] + '_' + metrics_df.index[metrics_df.index.get_loc(metrics_df.index[metrics_df[col].notna()][0])][1]
            self.log_message(f"Normalizing {col} with data mean: {data_mean} (from {loc_dataset}_{method})")
            if any(metric in col for metric in metrics_to_normalize):
                if data_mean != 0 and pd.notna(data_mean):
                    mean_df[col] = metrics_df[col] / data_mean
                else:
                    mean_df[col] = metrics_df[col]
            else:
                mean_df[col] = metrics_df[col]
        
        # T-tests
        ttest_results = []
        locations = sorted(list(set(metrics_df.index.get_level_values('Location'))))
        self.log_message(f"Unique locations for t-test: {locations}")
        
        for metric in METRICS:
            for method in METHODS:
                col = f"{metric}_{method}"
                if col not in metrics_df.columns:
                    self.log_message(f"Column {col} not found in metrics_df")
                    continue
                try:
                    gpm_values = []
                    chirps_values = []
                    for loc in locations:
                        gpm_val = metrics_df.loc[(loc, 'GPM'), col] if (loc, 'GPM') in metrics_df.index else np.nan
                        chirps_val = metrics_df.loc[(loc, 'CHIRPS'), col] if (loc, 'CHIRPS') in metrics_df.index else np.nan
                        if pd.notna(gpm_val) and pd.notna(chirps_val):
                            gpm_values.append(float(gpm_val))
                            chirps_values.append(float(chirps_val))
                    self.log_message(f"T-test for {col}: GPM={gpm_values}, CHIRPS={chirps_values}")
                    
                    if len(gpm_values) >= 2 and len(gpm_values) == len(chirps_values):
                        t_stat, p_value = ttest_rel(gpm_values, chirps_values)
                        mean_diff = np.mean(np.array(gpm_values) - np.array(chirps_values))
                        ttest_results.append({
                            'Metric': col,
                            'Mean_Diff (GPM - CHIRPS)': mean_diff,
                            't_stat': t_stat,
                            'p_value': p_value,
                            'Significant': 'Yes' if p_value < 0.05 else 'No'
                        })
                        self.log_message(f"T-test for {col}: Mean_Diff={mean_diff:.3f}, t_stat={t_stat:.3f}, p_value={p_value:.3f}, Significant={'Yes' if p_value < 0.05 else 'No'}")
                    else:
                        self.log_message(f"Skipping t-test for {col}: Insufficient valid pairs (GPM: {len(gpm_values)}, CHIRPS: {len(chirps_values)})")
                except Exception as e:
                    self.log_message(f"Error in t-test for {col}: {e}")
        
        # Save results
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                metrics_df.to_excel(writer, sheet_name='Raw_Metrics')
                stats.to_excel(writer, sheet_name='Statistics')
                mean_df.to_excel(writer, sheet_name='Mean_Metrics')
                storm_thresholds_df.to_excel(writer, sheet_name='Storm_Thresholds')
                pd.DataFrame(ttest_results).to_excel(writer, sheet_name='TTest_Results')
                if not data_stats_df.empty:
                    data_stats_df.to_excel(writer, sheet_name='Data_Means')
                else:
                    self.log_message("Data_Means DataFrame is empty; skipping sheet.")
            
            metrics_df.reset_index().to_csv(os.path.join(input_dir, f"Metrics_Summary_{agg_type}.csv"), index=False)
            self.log_message(f"Results saved to {output_file} and {os.path.join(input_dir, f'Metrics_Summary_{agg_type}.csv')}")
            messagebox.showinfo("Success", f"Metrics processing completed. Results saved to {output_file}")
        except Exception as e:
            self.log_message(f"Error saving results: {e}")
            messagebox.showerror("Error", f"Failed to save results: {e}")

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = MetricsProcessorGUI(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"GUI initialization failed: {e}")
        messagebox.showerror("Error", f"GUI initialization failed: {e}")