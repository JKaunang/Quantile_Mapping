'''
Rainfall Data Sorter Advanced 
By: James Albert Kaunang, Improved by Grok
Last Updated: June 10, 2025
Changelog:
- Added 'Data Completeness (%)' column to report output for percentage of non-missing data days per month
- Added 'Average Rainfall (mm)' column to report output for monthly mean rainfall (including zero days)
- Added 'Total Rainfall (mm)' column to report output for monthly rainfall accumulation
- Added dynamic date column selection via GUI dropdown
- Modified read_input_file to use user-specified date column
- Updated find_header_row to focus on month columns only
- Adjusted run_script to pass user-selected date column
- Updated configuration saving/loading to include date column
- Fixed bracket-related syntax errors in report_data_statistics and run_script
- Fixed syntax errors in read_input_file, run_script, and RainfallApp.run
- Allowed processing of months with no valid numeric data (e.g., all missing due to broken gauge)
- Sorted months in Report sheet chronologically (Jan, Feb, Mar, ..., Dec) instead of alphabetically
- Fixed syntax errors in extract_year_from_filename and find_header_row
- Fixed missing value counts to only include valid days per month (e.g., 1 – 31 for January)
- Added option to merge batch-processed outputs into a single file with timestamped name
- Added blank cells as missing values (treated as NaN alongside "-" and "9999")
- Added support for .xls files
- Fixed missing value reporting to treat "-" and "9999" as missing, preserve 0 as valid
- Added overwrite warning, extracted years display, progress details, input validation
- Fixed progress bar
- Reverted Date-Month-Year to datetime, unified output folder and naming
- Extracted year from filename, added batch processing
- Adjusted GUI (darkly theme, new theme options, plots off by default)

This code sorts rainfall data from Excel (.xls, .xlsx) or CSV files, generates a recap and report with missing data, statistics, total rainfall, average rainfall, and data completeness percentage per month, and supports single-file and batch processing with year extracted from filenames. Batch mode can merge outputs into one file. Missing values are counted only for valid days per month, months in the Report are sorted chronologically, and months with no valid data are allowed. The date column is dynamically selected via the GUI.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import calendar
import logging
import tkinter as tk
from tkinter import messagebox, filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap.tooltip import ToolTip
from typing import List, Dict, Optional, Tuple
import plotly.express as px 
import json
from functools import lru_cache
import re
from datetime import datetime

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define chronological month order
MONTH_ORDER = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

def extract_year_from_filename(filename: str) -> int:
    """
    Extract a four-digit year from the filename using regex.
    Returns a default year (2000) if no year is found.
    """
    try:
        match = re.search(r'\d{4}', filename)
        if match:
            year = int(match.group(0))
            logging.info(f"Extracted year {year} from filename {filename}")
            return year
        logging.warning(f"No year found in {filename}, using default year 2000")
        return 2000
    except Exception as e:
        logging.warning(f"Error extracting year from {filename}: {e}, using default year 2000")
        return 2000

def find_header_row(df: pd.DataFrame) -> int:
    """
    Find the row index where month columns are located.
    """
    valid_months = {month.lower() for month in calendar.month_abbr[1:]}
    
    for i, row in df.iterrows():
        try:
            row_values = {str(val).lower() for val in row.values if pd.notna(val)}
            logging.debug(f"Checking row {i}: {row_values}")
            if any(month in row_values for month in valid_months):
                logging.info(f"Header row found at index {i}")
                return i
        except Exception as e:
            logging.error(f"Error checking row {i}: {e}")
            continue
    
    logging.error("Header row with month columns not found")
    return -1

def read_input_file(file_path: str, date_column: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Read and validate the input Excel (.xls, .xlsx) or CSV file. Returns (DataFrame, error_message).
    Uses user-specified date_column if provided, otherwise tries to infer.
    Filters rows with valid Date values based on month-specific day ranges.
    Allows months with no valid numeric data (all NaN after cleaning).
    """
    logging.info(f"Reading and validating input file: {file_path}")
    try:
        if not file_path.endswith(('.xls', '.xlsx', '.csv')):
            return None, "Unsupported file format. Use .xls, .xlsx, or .csv"

        # Read file
        if file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path, keep_default_na=False)
        else:
            df = pd.read_csv(file_path, keep_default_na=False)
        
        # Find header
        header_row = find_header_row(df)
        if header_row == -1:
            return None, "Header row with month columns not found"
        
        df.columns = df.iloc[header_row]
        df = df.drop(range(header_row + 1)).reset_index(drop=True)
        
        # Validate date column
        if date_column and date_column in df.columns:
            df.rename(columns={date_column: 'Date'}, inplace=True)
        else:
            # Fallback to inferring date column
            valid_date_columns = {'date', 'day', 'timestamp'}
            date_col = next((col for col in df.columns if str(col).lower() in valid_date_columns), None)
            if not date_col:
                return None, "No valid date column found. Please select a date column."
            df.rename(columns={date_col: 'Date'}, inplace=True)
        
        # Validate month columns
        valid_months = set(calendar.month_abbr[1:])
        month_columns = [col for col in df.columns[1:] if str(col).capitalize() in valid_months]
        invalid_columns = [col for col in df.columns[1:] if str(col).capitalize() not in valid_months]
        if invalid_columns:
            logging.warning(f"Ignoring invalid or empty columns: {invalid_columns}")
        if not month_columns:
            return None, "No valid month columns (e.g., JAN, FEB) found"
        
        df = df[['Date'] + month_columns]
        
        # Convert Date to numeric and validate
        df['Date'] = pd.to_numeric(df['Date'], errors='coerce').astype('Int64')
        if df['Date'].isnull().all():
            return None, "All 'Date' values are invalid or non-numeric"
        
        # Filter valid Date values based on month-specific day ranges
        year = extract_year_from_filename(file_path)
        valid_rows = []
        for index, row in df.iterrows():
            date = row['Date']
            if pd.isna(date):
                continue
            for month in month_columns:
                month_num = list(calendar.month_abbr).index(month.capitalize())
                _, max_days = calendar.monthrange(year, month_num)
                if 1 <= date <= max_days:
                    valid_rows.append(index)
                    break  # Valid for at least one month, keep row
        if not valid_rows:
            return None, "No rows with valid Date values for any month"
        df = df.loc[valid_rows].reset_index(drop=True)
        
        # Validate numeric data in month columns
        temp_df = df.copy()
        temp_df[month_columns] = temp_df[month_columns].replace(['-', '9999', ''], np.nan)
        valid_months_found = False
        for col in month_columns:
            try:
                temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
                if not temp_df[col].isnull().all():
                    valid_months_found = True
                else:
                    logging.warning(f"Month '{col}' in {file_path} has no valid numeric data after cleaning")
            except Exception:
                return None, f"Column '{col}' contains invalid data (must be numeric)"
        
        if not valid_months_found:
            return None, "All month columns contain no valid numeric data after cleaning"
        
        logging.debug(f"Valid columns: {df.columns.tolist()}, {len(df)} rows after filtering")
        return df, None
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return None, f"Error reading file: {str(e)}"

@lru_cache(maxsize=128)
def validate_date(year: int, month: str, day: int) -> bool:
    """
    Validate if a given day exists in the specified month and year.
    """
    try:
        month_number = list(calendar.month_abbr).index(month.capitalize())
        _, last_day = calendar.monthrange(year, month_number)
        return day <= last_day
    except ValueError:
        return False

def process_data(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Process the input DataFrame to rearrange and validate dates into a long format.
    Outputs Date-Month-Year in a format suitable for datetime parsing (e.g., 2002-Jan-01).
    """
    logging.info("Processing data with vectorized operations")
    month_columns = df.columns[1:]
    melted_df = pd.melt(df, id_vars=['Date'], value_vars=month_columns, 
                        var_name='Month', value_name='Rainfall Magnitude')
    
    melted_df['Date-Month-Year'] = melted_df.apply(
        lambda row: f"{year}-{row['Month'].capitalize()}-{int(row['Date']):02d}" 
        if pd.notna(row['Date']) and validate_date(year, row['Month'].capitalize(), row['Date']) else None, 
        axis=1)
    
    melted_df = melted_df.dropna(subset=['Date-Month-Year'])
    logging.debug("Data processed successfully")
    return melted_df[['Date-Month-Year', 'Rainfall Magnitude']]

def report_missing_data(df: pd.DataFrame, file_name: str = '') -> pd.DataFrame:
    """
    Report missing data (NaN after replacing '-', 9999, blanks) and data completeness for valid days only.
    Optionally include file name for merged reports. Sort months chronologically.
    """
    if df.empty:
        logging.warning(f"Empty DataFrame for {file_name}, no missing data report generated")
        return pd.DataFrame(columns=['File', 'Month', 'Missing Values', 'Missing Dates', 'Data Completeness (%)'])
    
    year = extract_year_from_filename(file_name) if file_name else 2000
    missing_data_report = {
        'File': [],
        'Month': [],
        'Missing Values': [],
        'Missing Dates': [],
        'Data Completeness (%)': []
    }
    
    # Process months in chronological order
    month_columns = [col for col in df.columns[1:] if col.upper() in MONTH_ORDER]
    month_columns.sort(key=lambda x: MONTH_ORDER.index(x.upper()))
    
    for col in month_columns:
        month_num = list(calendar.month_abbr).index(col.capitalize())
        _, max_days = calendar.monthrange(year, month_num)
        valid_dates = df['Date'].apply(lambda x: pd.notna(x) and 1 <= x <= max_days)
        valid_data = df.loc[valid_dates, col]
        
        valid_data = pd.to_numeric(valid_data, errors='coerce')
        missing_count = valid_data.isnull().sum()
        missing_data_report['File'].append(file_name)
        missing_data_report['Month'].append(col)
        missing_data_report['Missing Values'].append(missing_count)
        missing_dates = df['Date'][valid_dates & valid_data.isna()].astype(str).tolist()
        missing_data_report['Missing Dates'].append(', '.join(missing_dates))
        # Calculate data completeness percentage
        valid_days = valid_dates.sum()
        non_missing_days = valid_days - missing_count
        completeness = (non_missing_days / max_days * 100) if max_days > 0 else 0.0
        missing_data_report['Data Completeness (%)'].append(round(completeness, 2))
    
    report_df = pd.DataFrame(missing_data_report)
    logging.debug(f"Missing data report for {file_name}: {report_df.to_dict()}")
    return report_df

def report_data_statistics(df: pd.DataFrame, year: int, file_name: str = '') -> pd.DataFrame:
    """
    Report statistics including maximum, minimum, number of rainy days (value > 0), total rainfall, and average rainfall.
    Only consider valid days for each month. Sort months chronologically.
    """
    if df.empty:
        logging.warning(f"Empty DataFrame for {file_name}, no statistics report generated")
        return pd.DataFrame(columns=['File', 'Month', 'Maximum', 'Max Date', 'Minimum', 'Min Date', 
                                    'Number of Rainy Day(s) (value > 0)', 'Total Rainfall (mm)', 'Average Rainfall (mm)'])
    
    stats = {
        'File': [],
        'Month': [],
        'Maximum': [],
        'Max Date': [],
        'Minimum': [],
        'Min Date': [],
        'Number of Rainy Day(s) (value > 0)': [],
        'Total Rainfall (mm)': [],
        'Average Rainfall (mm)': []
    }
    
    # Process months in chronological order
    month_columns = [col for col in df.columns[1:] if col.upper() in MONTH_ORDER]
    month_columns.sort(key=lambda x: MONTH_ORDER.index(x.upper()))
    
    for column in month_columns:
        month_num = list(calendar.month_abbr).index(column.capitalize())
        _, max_days = calendar.monthrange(year, month_num)
        valid_dates = df['Date'].apply(lambda x: pd.notna(x) and 1 <= x <= max_days)
        valid_data = df[valid_dates].copy()
        
        if not valid_data.empty:
            stats['File'].append(file_name)
            stats['Month'].append(column)
            valid_data[column] = pd.to_numeric(valid_data[column], errors='coerce')
            max_value = valid_data[column].max()
            min_value = valid_data[column].min()
            
            max_date = None
            if pd.notna(max_value):
                max_dates = valid_data['Date'][valid_data[column] == max_value]
                max_date = max_dates.values[0] if not max_dates.empty else None
            
            min_date = None
            if pd.notna(min_value):
                non_zero = valid_data[valid_data[column] > 0][column]
                if not non_zero.empty:
                    min_value = non_zero.min()
                    min_dates = valid_data['Date'][valid_data[column] == min_value]
                    min_date = min_dates.values[0] if not min_dates.empty else None
                else:
                    min_date = valid_data['Date'].iloc[0] if not valid_data.empty else None
            
            stats['Maximum'].append(max_value)
            stats['Max Date'].append(max_date)
            stats['Minimum'].append(min_value)
            stats['Min Date'].append(min_date)
            stats['Number of Rainy Day(s) (value > 0)'].append((valid_data[column] > 0).sum())
            # Calculate total rainfall (sum of valid, non-NaN values)
            total_rainfall = valid_data[column].sum() if not valid_data[column].isnull().all() else 0.0
            stats['Total Rainfall (mm)'].append(round(total_rainfall, 2))
            # Calculate average rainfall (mean of valid, non-NaN values, including zeros)
            avg_rainfall = valid_data[column].mean() if not valid_data[column].isnull().all() else 0.0
            stats['Average Rainfall (mm)'].append(round(avg_rainfall, 2))
    
    report_df = pd.DataFrame(stats)
    logging.debug(f"Statistics report for {file_name}: {report_df.to_dict()}")
    return report_df

def merge_reports(missing_reports: List[pd.DataFrame], stats_reports: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge missing data and statistics reports into one DataFrame.
    Sort months chronologically using MONTH_ORDER.
    """
    merged_missing = pd.concat(missing_reports, ignore_index=True) if missing_reports else pd.DataFrame()
    merged_stats = pd.concat(stats_reports, ignore_index=True) if stats_reports else pd.DataFrame()
    
    if not merged_missing.empty and not merged_stats.empty:
        merged_report = pd.merge(merged_missing, merged_stats, on=['File', 'Month'], how='outer')
    elif not merged_missing.empty:
        merged_report = merged_missing
    elif not merged_stats.empty:
        merged_report = merged_stats
    else:
        merged_report = pd.DataFrame()
    
    if not merged_report.empty:
        # Add MonthOrder column for sorting
        merged_report['MonthOrder'] = merged_report['Month'].str.upper().map(lambda x: MONTH_ORDER.index(x) if x in MONTH_ORDER else len(MONTH_ORDER))
        merged_report = merged_report.sort_values(['File', 'MonthOrder']).drop(columns=['MonthOrder'])
    
    logging.info(f"Merged reports with {len(merged_report)} rows")
    return merged_report

def save_and_plot_data(output_df: pd.DataFrame, output_file: str, combined_report: pd.DataFrame, 
                      plot_figure: bool, export_format: str = 'xlsx') -> bool:
    """
    Save the processed data and plot if required. Checks for file overwrite.
    Returns True if saved, False if user cancels.
    """
    if os.path.exists(output_file):
        response = messagebox.askyesnocancel("Overwrite File", 
                                            f"File {os.path.basename(output_file)} already exists. Overwrite?")
        if response is False:  # No
            logging.info(f"User chose not to overwrite {output_file}")
            return False
        elif response is None:  # Cancel
            logging.info(f"User cancelled saving {output_file}")
            return False
    
    try:
        if export_format == 'xlsx':
            with pd.ExcelWriter(output_file) as writer:
                output_df.to_excel(writer, sheet_name='Processed Data', index=False)
                combined_report.to_excel(writer, sheet_name='Report', index=False)
        else:
            output_df.to_csv(output_file, index=False)
            combined_report.to_csv(output_file.replace('.csv', '_report.csv'), index=False)
        logging.info(f"Data saved to {output_file}")
        
        if plot_figure:
            output_folder = os.path.dirname(output_file)
            plot_data(output_df, output_folder)
        return True
    except Exception as e:
        logging.error(f"Error saving or plotting data to {output_file}: {e}")
        raise

def plot_interactive(output_df: pd.DataFrame, output_folder: str) -> None:
    """
    Create an interactive Plotly line plot.
    """
    output_df['Date-Month-Year'] = pd.to_datetime(output_df['Date-Month-Year'])
    fig = px.line(output_df, x='Date-Month-Year', y='Rainfall Magnitude', 
                  title='Rainfall Over Time (Interactive Plot)',
                  labels={'Date-Month-Year': 'Date', 'Rainfall Magnitude': 'Rainfall (mm)'},
                  template='plotly_dark')
    fig.update_traces(mode='lines+markers', marker=dict(size=8))
    fig.update_xaxes(tickformat="%Y-%m-%d")
    fig.show()

def plot_static_line(output_df: pd.DataFrame, output_folder: str) -> None:
    """
    Create a static Matplotlib line plot.
    """
    output_df['Date-Month-Year'] = pd.to_datetime(output_df['Date-Month-Year'])
    plt.figure(figsize=(14, 6))
    sns.lineplot(x='Date-Month-Year', y='Rainfall Magnitude', data=output_df, marker='o', color='blue')
    plt.title('Rainfall Over Time (Line Plot)')
    plt.xlabel('Date')
    plt.ylabel('Rainfall Magnitude (mm)')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'rainfall_line_plot.png'))
    plt.close()

def plot_static_bar(output_df: pd.DataFrame, output_folder: str) -> None:
    """
    Create a static Matplotlib bar plot.
    """
    output_df['Date-Month-Year'] = pd.to_datetime(output_df['Date-Month-Year'])
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Date-Month-Year', y='Rainfall Magnitude', data=output_df, color='skyblue')
    plt.title('Rainfall Over Time (Bar Plot)')
    plt.xlabel('Date')
    plt.ylabel('Rainfall Magnitude (mm)')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'rainfall_bar_plot.png'))
    plt.close()

def plot_data(output_df: pd.DataFrame, output_folder: str) -> None:
    """
    Generate interactive and static plots.
    """
    plot_interactive(output_df, output_folder)
    plot_static_line(output_df, output_folder)
    plot_static_bar(output_df, output_folder)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame by replacing invalid values ('-', 9999, blanks) with NaN and ensuring non-negative rainfall.
    """
    df = df.replace(['-', '9999', ''], np.nan)
    month_columns = df.columns[1:]
    df[month_columns] = df[month_columns].apply(lambda x: pd.to_numeric(x, errors='coerce'))
    df[month_columns] = df[month_columns].clip(lower=0)
    logging.debug(f"Cleaned data, shape: {df.shape}")
    return df

def run_script(input_files: List[str], output_dir: str, plot_figure: bool, export_format: str = 'xlsx', 
               merge_outputs: bool = False, progress_callback: Optional[callable] = None, 
               date_column: Optional[str] = None) -> Tuple[int, List[str]]:
    """
    Run the script for one or more input files, processing each and saving to the output directory.
    If merge_outputs is True, combine all data into one file. Calls progress_callback with (processed_files, current_file).
    Uses user-specified date_column if provided.
    """
    success_count = 0
    errors = []
    all_output_dfs = []
    all_missing_reports = []
    all_stats_reports = []

    for i, input_file in enumerate(input_files):
        logging.info(f"Processing file: {input_file}")
        try:
            # Extract year from filename
            year = extract_year_from_filename(os.path.basename(input_file))
            # Generate output filename for separate outputs
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_processed.{export_format}")

            # Read and validate file with user-specified date column
            df, error = read_input_file(input_file, date_column=date_column)
            if df is None:
                raise ValueError(error)

            df = clean_data(df)
            if df['Date'].isnull().any():
                logging.warning(f"Some dates in {input_file} could not be converted to integers and were set to NaN")

            # Generate reports with file name for merging
            missing_data_report = report_missing_data(df, os.path.basename(input_file))
            data_stats_report = report_data_statistics(df, year, os.path.basename(input_file))
            combined_report = merge_reports([missing_data_report], [data_stats_report])
            output_df = process_data(df, year)

            # Convert Date-Month-Year to datetime
            output_df['Date-Month-Year'] = pd.to_datetime(output_df['Date-Month-Year'], errors='coerce')
            output_df = output_df.sort_values(by='Date-Month-Year')

            if merge_outputs:
                all_output_dfs.append(output_df)
                all_missing_reports.append(missing_data_report)
                all_stats_reports.append(data_stats_report)
                success_count += 1
                logging.info(f"Collected data from {input_file} for merging")
            else:
                # Save data and plots for separate outputs
                if not save_and_plot_data(output_df, output_file, combined_report, plot_figure, export_format):
                    logging.info(f"Skipped processing {input_file} due to user choice")
                    errors.append(f"Skipped {input_file}: User chose not to overwrite or cancelled")
                    continue
                
                success_count += 1
                logging.info(f"Successfully processed {input_file} to {output_file} with year {year}")

            # Update progress
            if progress_callback:
                progress_callback(i + 1, input_file)
        except Exception as e:
            error_msg = f"Error processing {input_file}: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)

    if merge_outputs and all_output_dfs:
        try:
            # Merge data and reports
            merged_df = merge_processed_data(all_output_dfs)
            merged_report = merge_reports(all_missing_reports, all_stats_reports)
            
            # Generate timestamped output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = os.path.join(output_dir, f"batch_processed_{timestamp}.{export_format}")

            # Save merged data
            if not save_and_plot_data(merged_df, output_file, merged_report, plot_figure, export_format):
                logging.info(f"Skipped saving merged output due to user choice")
                errors.append(f"Skipped merged output: User chose not to overwrite or cancelled")
            else:
                logging.info(f"Successfully saved merged output to {output_file}")
        except Exception as e:
            error_msg = f"Error merging outputs: {str(e)}"
            logging.error(error_msg)
            errors.append(error_msg)

    return success_count, errors

def merge_processed_data(data_frames: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple processed DataFrames into one, sorted by Date-Month-Year.
    """
    if not data_frames:
        logging.warning("No DataFrames to merge")
        return pd.DataFrame()
    merged_df = pd.concat(data_frames, ignore_index=True)
    merged_df['Date-Month-Year'] = pd.to_datetime(merged_df['Date-Month-Year'])
    merged_df = merged_df.sort_values(by='Date-Month-Year').reset_index(drop=True)
    logging.info(f"Merged {len(data_frames)} DataFrames with {len(merged_df)} rows")
    return merged_df

class RainfallApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Rainfall Data Processor")
        self.style = ttk.Style('darkly')
        
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(N, S, E, W))
        
        # Theme selection
        ttk.Label(self.main_frame, text="Theme:", font=('Helvetica', 10)).grid(row=0, column=0, padx=10, pady=5, sticky=W)
        self.theme_var = ttk.StringVar(value='darkly')
        theme_combo = ttk.Combobox(self.main_frame, textvariable=self.theme_var, 
                                  values=['flatly', 'darkly', 'superhero', 'litera', 'cyborg', 'yeti'], state='readonly', width=10)
        theme_combo.grid(row=0, column=1, padx=10, pady=5, sticky=W)
        theme_combo.bind('<<ComboboxSelected>>', self.change_theme)
        self.theme_tooltip = ToolTip(theme_combo, text="Select the GUI theme")

        # Batch processing option
        self.batch_var = ttk.BooleanVar(value=False)
        batch_check = ttk.Checkbutton(self.main_frame, text="Batch Processing", variable=self.batch_var, 
                                     command=self.toggle_batch_mode, bootstyle=INFO)
        batch_check.grid(row=1, column=1, padx=10, pady=5, sticky=W)
        self.batch_tooltip = ToolTip(batch_check, text="Check to process multiple files at once (year extracted from filename)")

        # Merge outputs option
        self.merge_var = ttk.BooleanVar(value=False)
        self.merge_check = ttk.Checkbutton(self.main_frame, text="Merge Outputs", variable=self.merge_var, 
                                          bootstyle=INFO, state='disabled')
        self.merge_check.grid(row=1, column=2, padx=10, pady=5, sticky=W)
        self.merge_tooltip = ToolTip(self.merge_check, text="Check to combine all batch outputs into one file")

        # Input file/directory selection
        self.input_label = ttk.Label(self.main_frame, text="Input File:", font=('Helvetica', 10))
        self.input_label.grid(row=2, column=0, padx=10, pady=5, sticky=W)
        self.input_entry = ttk.Entry(self.main_frame, width=50)
        self.input_entry.grid(row=2, column=1, padx=10, pady=5, sticky=EW)
        self.input_btn = ttk.Button(self.main_frame, text="Browse", command=self.browse_input, bootstyle=PRIMARY)
        self.input_btn.grid(row=2, column=2, padx=10, pady=5)
        self.input_tooltip = ToolTip(self.input_btn, text="Select an Excel/CSV file (year extracted from filename)")

        # Date column selection
        ttk.Label(self.main_frame, text="Date Column:", font=('Helvetica', 10)).grid(row=2, column=3, padx=10, pady=5, sticky=W)
        self.date_column_var = ttk.StringVar()
        self.date_column_combo = ttk.Combobox(self.main_frame, textvariable=self.date_column_var, state='readonly', width=15)
        self.date_column_combo.grid(row=2, column=4, padx=10, pady=5, sticky=W)
        self.date_column_tooltip = ToolTip(self.date_column_combo, text="Select the column representing dates")
        self.date_column_combo['values'] = []  # Initially empty
        self.date_column_combo.grid_remove()  # Hide until file is selected

        # Display extracted years
        self.files_label = ttk.Label(self.main_frame, text="Selected Files and Years:", font=('Helvetica', 10))
        self.files_label.grid(row=3, column=0, padx=10, pady=5, sticky=W)
        self.files_listbox = tk.Listbox(self.main_frame, height=5, width=50)
        self.files_listbox.grid(row=3, column=1, columnspan=2, padx=10, pady=5, sticky=EW)
        self.files_tooltip = ToolTip(self.files_listbox, text="Shows selected files and their extracted years")

        # Output directory selection
        self.output_label = ttk.Label(self.main_frame, text="Output Directory:", font=('Helvetica', 10))
        self.output_label.grid(row=4, column=0, padx=10, pady=5, sticky=W)
        self.output_entry = ttk.Entry(self.main_frame, width=50)
        self.output_entry.grid(row=4, column=1, padx=10, pady=5, sticky=EW)
        self.output_btn = ttk.Button(self.main_frame, text="Browse", command=self.browse_output, bootstyle=PRIMARY)
        self.output_btn.grid(row=4, column=2, padx=10, pady=5)
        self.output_tooltip = ToolTip(self.output_btn, text="Select directory to save processed files")

        # Plot figure option
        self.plot_figure_var = ttk.BooleanVar(value=False)
        plot_check = ttk.Checkbutton(self.main_frame, text="Generate Plots", variable=self.plot_figure_var, bootstyle=INFO)
        plot_check.grid(row=5, column=1, padx=10, pady=5, sticky=W)
        self.plot_tooltip = ToolTip(plot_check, text="Check to generate interactive and static plots")

        # Export format
        ttk.Label(self.main_frame, text="Export Format:", font=('Helvetica', 10)).grid(row=6, column=0, padx=10, pady=5, sticky=W)
        self.export_format_var = ttk.StringVar(value='xlsx')
        export_combo = ttk.Combobox(self.main_frame, textvariable=self.export_format_var, 
                                    values=['xlsx', 'csv'], state='readonly', width=10)
        export_combo.grid(row=6, column=1, padx=10, pady=5, sticky=W)
        self.export_tooltip = ToolTip(export_combo, text="Select output format (Excel or CSV)")

        # Run button
        run_btn = ttk.Button(self.main_frame, text="Process Data", command=self.run, bootstyle=SUCCESS)
        run_btn.grid(row=7, column=1, padx=10, pady=20, sticky=EW)
        self.run_tooltip = ToolTip(run_btn, text="Start processing the rainfall data")

        # Status label and progress bar
        self.status_label = ttk.Label(self.main_frame, text="Ready", font=('Helvetica', 10, 'italic'), bootstyle=INFO)
        self.status_label.grid(row=8, column=0, columnspan=3, padx=10, pady=5)
        self.progress = ttk.Progressbar(self.main_frame, mode='determinate', bootstyle=INFO)
        self.progress.grid(row=9, column=0, columnspan=3, padx=10, pady=5, sticky=EW)
        self.progress.grid_remove()

        # Load/Save configuration buttons
        ttk.Button(self.main_frame, text="Load Config", command=self.load_config, bootstyle=SECONDARY).grid(row=10, column=0, padx=10, pady=10)
        ttk.Button(self.main_frame, text="Save Config", command=self.save_config, bootstyle=SECONDARY).grid(row=10, column=1, padx=10, pady=10, sticky=W)

        # Configure grid weights
        self.main_frame.columnconfigure(1, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Initialize batch mode
        self.toggle_batch_mode()

    def change_theme(self, event=None):
        """Change the GUI theme."""
        self.style.theme_use(self.theme_var.get())
        self.status_label.config(text=f"Theme changed to {self.theme_var.get()}")

    def toggle_batch_mode(self):
        """Toggle between single-file and batch processing modes."""
        if self.batch_var.get():
            self.input_label.config(text="Input Files:")
            self.output_label.config(text="Output Directory:")
            self.input_tooltip.text = "Select multiple Excel/CSV files (year extracted from filename)"
            self.output_tooltip.text = "Select directory to save processed files"
            self.merge_check.config(state='normal')
        else:
            self.input_label.config(text="Input File:")
            self.output_label.config(text="Output Directory:")
            self.input_tooltip.text = "Select an Excel/CSV file (year extracted from filename)"
            self.output_tooltip.text = "Select directory to save processed file"
            self.merge_check.config(state='disabled')
            self.merge_var.set(False)
        self.input_entry.delete(0, tk.END)
        self.output_entry.delete(0, tk.END)
        self.files_listbox.delete(0, tk.END)
        self.date_column_combo['values'] = []
        self.date_column_combo.grid_remove()
        self.date_column_var.set('')
        self.status_label.config(text="Ready", bootstyle=INFO)

    def browse_input(self):
        """Open a file dialog to select input file(s), display extracted years, and populate date column selector."""
        try:
            self.files_listbox.delete(0, tk.END)
            self.date_column_combo['values'] = []
            self.date_column_combo.grid_remove()
            self.date_column_var.set('')

            if self.batch_var.get():
                file_paths = filedialog.askopenfilenames(filetypes=[("All files", "*.xls *.xlsx *.csv")])
                if file_paths:
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, "; ".join(file_paths))
                    # Read the first file to get column names
                    df, error = read_input_file(file_paths[0], date_column=None)
                    if df is not None:
                        columns = df.columns.tolist()
                        self.date_column_combo['values'] = columns
                        self.date_column_combo.grid()
                        if columns:
                            self.date_column_var.set(columns[0])  # Default to first column
                        for path in file_paths:
                            year = extract_year_from_filename(os.path.basename(path))
                            self.files_listbox.insert(tk.END, f"{os.path.basename(path)}: {year}")
                        self.status_label.config(text=f"{len(file_paths)} input files selected", bootstyle=INFO)
                    else:
                        self.status_label.config(text=f"Error reading first file: {error}", bootstyle=WARNING)
                        messagebox.showerror("Error", error)
                else:
                    self.status_label.config(text="No input files selected", bootstyle=WARNING)
            else:
                file_path = filedialog.askopenfilename(filetypes=[("All files", "*.xls *.xlsx *.csv")])
                if file_path:
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, file_path)
                    df, error = read_input_file(file_path, date_column=None)
                    if df is not None:
                        columns = df.columns.tolist()
                        self.date_column_combo['values'] = columns
                        self.date_column_combo.grid()
                        if columns:
                            self.date_column_var.set(columns[0])  # Default to first column
                        year = extract_year_from_filename(os.path.basename(file_path))
                        self.files_listbox.insert(tk.END, f"{os.path.basename(file_path)}: {year}")
                        self.status_label.config(text=f"Input file selected: {os.path.basename(file_path)}", bootstyle=INFO)
                    else:
                        self.status_label.config(text=f"Error reading file: {error}", bootstyle=WARNING)
                        messagebox.showerror("Error", error)
                else:
                    self.status_label.config(text="No input file selected", bootstyle=WARNING)
        except Exception as e:
            logging.error(f"Error opening input file dialog: {e}")
            self.status_label.config(text="Error accessing file dialog", bootstyle=DANGER)
            messagebox.showerror("Error", f"Error accessing file dialog: {e}")

    def browse_output(self):
        """Open a dialog to select output directory."""
        try:
            dir_path = filedialog.askdirectory()
            if dir_path:
                self.output_entry.delete(0, tk.END)
                self.output_entry.insert(0, dir_path)
                self.status_label.config(text=f"Output directory selected: {os.path.basename(dir_path)}", bootstyle=INFO)
            else:
                self.status_label.config(text="No output directory selected", bootstyle=WARNING)
        except Exception as e:
            logging.error(f"Error opening output directory dialog: {e}")
            self.status_label.config(text="Error accessing file dialog", bootstyle=DANGER)
            messagebox.showerror("Error", f"Error accessing file dialog: {e}")

    def run(self):
        """Run the script with the provided inputs."""
        input_text = self.input_entry.get()
        output_dir = self.output_entry.get()
        plot_figure = self.plot_figure_var.get()
        export_format = self.export_format_var.get()
        is_batch = self.batch_var.get()
        merge_outputs = self.merge_var.get() and is_batch
        date_column = self.date_column_var.get() if self.date_column_var.get() else None

        if not input_text or not output_dir:
            self.status_label.config(text="Error: Please provide all inputs", bootstyle=DANGER)
            messagebox.showerror("Error", "Please provide all inputs.")
            return
        if not date_column and is_batch:
            self.status_label.config(text="Error: Please select a date column for batch processing", bootstyle=DANGER)
            messagebox.showerror("Error", "Please select a date column for batch processing.")
            return

        self.status_label.config(text="Processing data...", bootstyle=INFO)
        self.progress.grid()
        self.progress['value'] = 0
        self.root.update()

        try:
            if is_batch:
                input_files = input_text.split("; ")
                total_files = len(input_files)
                self.progress['maximum'] = total_files
                
                def update_progress(processed_files, current_file):
                    self.progress['value'] = processed_files
                    self.status_label.config(text=f"Processing {processed_files}/{total_files}: {os.path.basename(current_file)}")
                    self.root.update()

                success_count, errors = run_script(input_files, output_dir, plot_figure, export_format, merge_outputs, update_progress, date_column=date_column)
                if errors:
                    error_msg = f"Processed {success_count}/{total_files} files. Errors:\n" + "\n".join(errors)
                    self.status_label.config(text="Batch processing completed with errors", bootstyle=WARNING)
                    messagebox.showwarning("Batch Processing", error_msg)
                else:
                    output_msg = f"Successfully processed {success_count}/{total_files} files"
                    if merge_outputs:
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        output_msg += f" into batch_processed_{timestamp}.{export_format}"
                    self.status_label.config(text=output_msg, bootstyle=SUCCESS)
                    messagebox.showinfo("Success", output_msg)
            else:
                self.progress['maximum'] = 1
                self.progress['value'] = 0
                
                def update_progress(processed_files, current_file):
                    self.progress['value'] = processed_files
                    self.status_label.config(text=f"Processing {os.path.basename(current_file)}")
                    self.root.update()

                success_count, errors = run_script([input_text], output_dir, plot_figure, export_format, False, update_progress, date_column=date_column)
                if errors:
                    self.status_label.config(text="Processing failed", bootstyle=DANGER)
                    messagebox.showerror("Error", errors[0])
                else:
                    base_name = os.path.splitext(os.path.basename(input_text))[0]
                    output_file = os.path.join(output_dir, f"{base_name}_processed.{export_format}")
                    self.status_label.config(text="Processing complete!", bootstyle=SUCCESS)
                    messagebox.showinfo("Success", f"Data processed and saved to {output_file}")
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}", bootstyle=DANGER)
            messagebox.showerror("Error", f"Error during processing: {e}")
        finally:
            self.progress.grid_remove()

    def load_config(self):
        """Load configuration from a JSON file."""
        try:
            config_file = filedialog.askopenfilename(filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if config_file:
                with open(config_file, 'r') as file:
                    config = json.load(file)
                    self.input_entry.delete(0, tk.END)
                    self.input_entry.insert(0, config.get('input_file', ''))
                    self.output_entry.delete(0, tk.END)
                    self.output_entry.insert(0, config.get('output_file', ''))
                    self.plot_figure_var.set(config.get('plot_figure', False))
                    self.export_format_var.set(config.get('export_format', 'xlsx'))
                    self.batch_var.set(config.get('batch_mode', False))
                    self.merge_var.set(config.get('merge_outputs', False) and config.get('batch_mode', False))
                    self.date_column_var.set(config.get('date_column', ''))
                    self.toggle_batch_mode()
                    if config.get('input_file'):
                        self.files_listbox.delete(0, tk.END)
                        input_files = config['input_file'].split("; ") if self.batch_var.get() else [config['input_file']]
                        for path in input_files:
                            year = extract_year_from_filename(os.path.basename(path))
                            self.files_listbox.insert(tk.END, f"{os.path.basename(path)}: {year}")
                        if config.get('date_column'):
                            self.date_column_combo['values'] = [config['date_column']]
                            self.date_column_combo.grid()
                    self.status_label.config(text=f"Configuration loaded from {os.path.basename(config_file)}", bootstyle=INFO)
            else:
                self.status_label.config(text="No config file selected", bootstyle=WARNING)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            self.status_label.config(text="Error accessing file dialog", bootstyle=DANGER)
            messagebox.showerror("Error", f"Error accessing file dialog: {e}")

    def save_config(self):
        """Save configuration to a JSON file."""
        config = {
            'input_file': self.input_entry.get(),
            'output_file': self.output_entry.get(),
            'plot_figure': self.plot_figure_var.get(),
            'export_format': self.export_format_var.get(),
            'batch_mode': self.batch_var.get(),
            'merge_outputs': self.merge_var.get(),
            'date_column': self.date_column_var.get()
        }
        try:
            config_file = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
            if config_file:
                with open(config_file, 'w') as file:
                    json.dump(config, file)
                self.status_label.config(text=f"Configuration saved to {os.path.basename(config_file)}", bootstyle=SUCCESS)
            else:
                self.status_label.config(text="No config file selected", bootstyle=WARNING)
        except Exception as e:
            logging.error(f"Error saving config: {e}")
            self.status_label.config(text="Error accessing file dialog", bootstyle=DANGER)
            messagebox.showerror("Error", f"Error accessing file dialog: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RainfallApp(root)
    root.mainloop()