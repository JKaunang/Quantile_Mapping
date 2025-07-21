'''
Rainfall GPM data Processor
By: James Albert Kaunang
Last Updated: June 10, 2025
Changelog:
- Added dynamic header row detection for CSV files.
- Added daily accumulation option for rainfall data.
- Added .xlsx output format support.

This script processes rainfall data from GPM (Global Precipitation Measurement) datasets, allowing users to 
convert 30-minute or daily rainfall data into daily accumulation with a threshold column. 
It supports both CSV and Excel output formats and includes a user-friendly GUI for easy interaction.
'''


import pandas as pd
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox
import os
import re
from datetime import datetime

# Constants for processing
FILL_VALUE = -9999.9  # Fill value for missing data
RAINFALL_THRESHOLD = 1.0  # Rainfall values below this (in mm) are set to 0 in threshold column
RAINFALL_MULTIPLIER = 2.0  # Multiplier for mean precipitation values
MAX_HEADER_SEARCH_ROWS = 10  # Maximum rows to search for header

def is_datetime_like(value):
    """Check if a value matches a datetime pattern (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS)."""
    if not isinstance(value, str):
        return False
    # Patterns for daily or 30-minute data
    patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$'  # YYYY-MM-DD HH:MM:SS
    ]
    return any(re.match(pattern, value.strip()) for pattern in patterns)

def is_numeric_like(value):
    """Check if a value is numeric or convertible to a number."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def find_header_row(file_path):
    """Search up to MAX_HEADER_SEARCH_ROWS for a header with at least one datetime-like and one numeric-like column."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = [next(file).strip() for _ in range(MAX_HEADER_SEARCH_ROWS) if file]
        
        for i, line in enumerate(lines):
            columns = [col.strip() for col in line.split(',')]
            if not columns:
                continue
            
            # Read the next row to check for data types (if available)
            next_row = None
            if i + 1 < len(lines):
                next_row = [col.strip() for col in lines[i + 1].split(',')]
                if len(next_row) != len(columns):
                    next_row = None  # Mismatch in column count
            
            # Check if this row is a header
            has_datetime = False
            has_numeric = False
            for j, col in enumerate(columns):
                # Check column name or next row's value for datetime
                if next_row and is_datetime_like(next_row[j]):
                    has_datetime = True
                # Check if next row's value is numeric
                if next_row and is_numeric_like(next_row[j]):
                    has_numeric = True
            
            if has_datetime and has_numeric:
                print(f"Detected header row {i}: {line}")
                return i
        
        raise ValueError(f"No valid header row found within the first {MAX_HEADER_SEARCH_ROWS} rows. Ensure the CSV has a header with datetime and numeric columns.")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {str(e)}")

def get_csv_columns(file_path, header_row):
    """Read the column names from the CSV's header row."""
    try:
        df = pd.read_csv(file_path, skiprows=header_row, nrows=0, encoding='utf-8', sep=None, engine='python')
        columns = df.columns.str.strip().tolist()
        print(f"Detected columns: {columns}")
        return columns
    except Exception as e:
        raise ValueError(f"Error reading CSV columns: {str(e)}")

def process_rainfall_data(file_path, output_file, output_format, data_frequency, time_column, rainfall_column):
    """Process rainfall data into daily accumulation with threshold column and generate report for xlsx."""
    try:
        # Validate output file path
        output_dir = os.path.dirname(output_file) or '.'
        if not os.access(output_dir, os.W_OK):
            raise ValueError(f"No write permission for output directory: {output_dir}")

        # Find the header row
        header_row = find_header_row(file_path)
        
        # Try reading CSV with common delimiters
        try:
            df = pd.read_csv(file_path, skiprows=header_row, encoding='utf-8', sep=None, engine='python')
        except Exception as e:
            print(f"Failed with sep=None, trying explicit delimiters: {str(e)}")
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(file_path, skiprows=header_row, encoding='utf-8', sep=sep)
                    print(f"Success with delimiter: {sep}")
                    break
                except:
                    continue
            else:
                raise ValueError(f"Failed to parse CSV with common delimiters (',', ';', '\t'). Error: {str(e)}")
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Print detected columns for debugging
        print(f"Detected columns after stripping whitespace: {list(df.columns)}")
        
        # Validate selected columns
        if time_column not in df.columns or rainfall_column not in df.columns:
            raise ValueError(f"Selected columns '{time_column}' or '{rainfall_column}' not found in CSV. Available columns: {list(df.columns)}")
        
        # Rename columns for consistency
        df = df.rename(columns={time_column: 'time', rainfall_column: 'rainfall'})
        print(f"Renamed columns to: {list(df.columns)}")
        
        # Check for missing or invalid values in the time column
        if df['time'].isnull().any():
            raise ValueError("The 'time' column contains missing (NaN) values.")
        
        # Parse the time column based on data frequency
        time_format = "%Y-%m-%d" if data_frequency == "daily" else "%Y-%m-%d %H:%M:%S"
        try:
            df['time'] = pd.to_datetime(df['time'], format=time_format, errors='raise')
        except Exception as e:
            raise ValueError(f"Failed to parse 'time' column with format {time_format}: {str(e)}")
        
        # Handle fill values in rainfall and convert to numeric
        df['rainfall'] = df['rainfall'].replace(FILL_VALUE, pd.NA)
        df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
        if df['rainfall'].isnull().any():
            print(f"Warning: Some rainfall values were invalid or equal to fill value ({FILL_VALUE}) and converted to NaN.")
        
        # Apply multiplier to rainfall values
        df['rainfall'] = df['rainfall'] * RAINFALL_MULTIPLIER
        print(f"Applied multiplier {RAINFALL_MULTIPLIER} to rainfall values")
        
        # Set the time as the index
        df.set_index('time', inplace=True)
        
        # Process rainfall data based on frequency
        if data_frequency == "30-minute":
            daily_rainfall = df['rainfall'].resample('D').sum()
        else:  # daily
            daily_rainfall = df['rainfall']
        
        # Apply rainfall threshold for the new column
        daily_rainfall_threshold = daily_rainfall.where(daily_rainfall >= RAINFALL_THRESHOLD, 0)
        
        # Create a DataFrame with both original and threshold-corrected results
        result_df = pd.DataFrame({
            'date': daily_rainfall.index,
            'daily_rainfall_mm': daily_rainfall.values,
            'daily_rainfall_threshold_mm': daily_rainfall_threshold.values
        })
        
        # Remove days with no data (if any)
        result_df = result_df.dropna()
        
        # Generate report data for xlsx output
        if output_format == 'xlsx':
            try:
                import openpyxl
            except ImportError:
                raise ImportError("openpyxl is required for xlsx output. Install it with 'pip install openpyxl'.")

            # Monthly totals
            monthly_totals = result_df.resample('ME', on='date')['daily_rainfall_mm'].sum().reset_index()
            monthly_totals['year_month'] = monthly_totals['date'].dt.strftime('%Y-%m')
            monthly_totals['year'] = monthly_totals['date'].dt.year
            
            # Yearly totals
            yearly_totals = result_df.resample('YE', on='date')['daily_rainfall_mm'].sum().reset_index()
            yearly_totals['year'] = yearly_totals['date'].dt.year
            
            # Report data: structured table by year
            report_data = []
            result_df['year'] = result_df['date'].dt.year
            years = sorted(result_df['year'].unique())
            
            for year in years:
                # Daily max/min
                year_daily = result_df[result_df['year'] == year]
                max_daily = year_daily.loc[year_daily['daily_rainfall_mm'].idxmax()] if not year_daily.empty else pd.Series({'daily_rainfall_mm': None, 'date': None})
                min_daily = year_daily[year_daily['daily_rainfall_mm'] > 0].loc[year_daily[year_daily['daily_rainfall_mm'] > 0]['daily_rainfall_mm'].idxmin()] if not year_daily[year_daily['daily_rainfall_mm'] > 0].empty else pd.Series({'daily_rainfall_mm': None, 'date': None})
                
                # Monthly max/min
                year_monthly = monthly_totals[monthly_totals['year'] == year]
                max_monthly = year_monthly.loc[year_monthly['daily_rainfall_mm'].idxmax()] if not year_monthly.empty else pd.Series({'daily_rainfall_mm': None, 'year_month': None})
                min_monthly = year_monthly[year_monthly['daily_rainfall_mm'] > 0].loc[year_monthly[year_monthly['daily_rainfall_mm'] > 0]['daily_rainfall_mm'].idxmin()] if not year_monthly[year_monthly['daily_rainfall_mm'] > 0].empty else pd.Series({'daily_rainfall_mm': None, 'year_month': None})
                
                # Yearly max/min (only one per year)
                year_total = yearly_totals[yearly_totals['year'] == year]
                yearly_value = year_total['daily_rainfall_mm'].iloc[0] if not year_total.empty else None
                
                report_data.append({
                    'Year': year,
                    'Max Daily (mm)': max_daily['daily_rainfall_mm'],
                    'Max Daily Date': max_daily['date'].strftime('%Y-%m-%d') if max_daily['date'] else None,
                    'Min Daily (mm)': min_daily['daily_rainfall_mm'],
                    'Min Daily Date': min_daily['date'].strftime('%Y-%m-%d') if min_daily['date'] else None,
                    'Max Monthly (mm)': max_monthly['daily_rainfall_mm'],
                    'Max Monthly Date': max_monthly['year_month'],
                    'Min Monthly (mm)': min_monthly['daily_rainfall_mm'],
                    'Min Monthly Date': min_monthly['year_month'],
                    'Max Yearly (mm)': yearly_value,
                    'Min Yearly (mm)': yearly_value
                })
            
            report_df = pd.DataFrame(report_data)
        
        # Save output
        print(f"Saving output to: {output_file} (format: {output_format})")
        if output_format == 'xlsx':
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                result_df.to_excel(writer, sheet_name='Daily Data', index=False)
                monthly_totals[['year_month', 'daily_rainfall_mm']].rename(columns={'year_month': 'Month', 'daily_rainfall_mm': 'Total Precipitation (mm)'}).to_excel(writer, sheet_name='Monthly Totals', index=False)
                yearly_totals[['year', 'daily_rainfall_mm']].rename(columns={'year': 'Year', 'daily_rainfall_mm': 'Total Precipitation (mm)'}).to_excel(writer, sheet_name='Yearly Totals', index=False)
                report_df.to_excel(writer, sheet_name='Report', index=False)
        else:  # csv
            result_df.to_csv(output_file, index=False)
        
        return True, f"Daily rainfall data saved to {output_file}"
    except Exception as e:
        return False, f"Error processing file: {str(e)}"

def create_gui():
    """Create the ttkbootstrap GUI for rainfall data processing."""
    root = ttk.Window(themename="darkly")
    root.title("Rainfall Data Processor for GPM Data")
    root.geometry("600x450")  # Height for dropdowns
    
    input_file = ttk.StringVar()
    output_file = ttk.StringVar()
    output_format = ttk.StringVar(value="xlsx")  # Default to xlsx
    data_frequency = ttk.StringVar(value="30-minute")
    time_column = ttk.StringVar()
    rainfall_column = ttk.StringVar()
    column_options = ttk.StringVar(value=["Select input file first"])
    
    def browse_input():
        """Open file dialog to select input CSV file and populate column dropdowns."""
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            input_file.set(file_path)
            try:
                header_row = find_header_row(file_path)
                columns = get_csv_columns(file_path, header_row)
                if not columns:
                    messagebox.showerror("Error", "No columns found in the CSV file.")
                    return
                column_options.set(columns)
                time_menu['menu'].delete(0, 'end')
                rainfall_menu['menu'].delete(0, 'end')
                for col in columns:
                    time_menu['menu'].add_command(label=col, command=lambda c=col: time_column.set(c))
                    rainfall_menu['menu'].add_command(label=col, command=lambda c=col: rainfall_column.set(c))
                time_column.set(columns[0] if columns else "")
                rainfall_column.set(columns[1] if len(columns) > 1 else columns[0] if columns else "")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to read CSV columns: {str(e)}")
                column_options.set(["Select input file first"])
                time_column.set("")
                rainfall_column.set("")
    
    def browse_output():
        """Open file dialog to select output file with appropriate extension."""
        file_types = [("CSV files", "*.csv"), ("Excel files", "*.xlsx")] if output_format.get() == "xlsx" else [("CSV files", "*.csv")]
        file_path = filedialog.asksaveasfilename(
            defaultextension=f".{output_format.get()}",
            filetypes=file_types,
            initialfile=f"daily_rainfall.{output_format.get()}"
        )
        if file_path:
            output_file.set(file_path)
    
    def process():
        """Process the rainfall data when the button is clicked."""
        if not input_file.get():
            messagebox.showerror("Error", "Please select an input CSV file.")
            return
        if not output_file.get():
            messagebox.showerror("Error", "Please select an output file.")
            return
        if not time_column.get() or not rainfall_column.get():
            messagebox.showerror("Error", "Please select both time and rainfall columns.")
            return
        
        success, message = process_rainfall_data(
            input_file.get(), output_file.get(), output_format.get(), 
            data_frequency.get(), time_column.get(), rainfall_column.get()
        )
        if success:
            messagebox.showinfo("Success", message)
        else:
            messagebox.showerror("Error", message)
    
    # GUI Layout
    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=BOTH, expand=True)
    
    # Input file selection
    ttk.Label(frame, text="Input CSV File:").grid(row=0, column=0, sticky=W, pady=5)
    ttk.Entry(frame, textvariable=input_file, width=50).grid(row=0, column=1, padx=5, pady=5)
    ttk.Button(frame, text="Browse", command=browse_input, bootstyle=PRIMARY).grid(row=0, column=2, padx=5)
    
    # Time column selection
    ttk.Label(frame, text="Time Column:").grid(row=1, column=0, sticky=W, pady=5)
    time_menu = ttk.OptionMenu(frame, time_column, *column_options.get())
    time_menu.grid(row=1, column=1, sticky=W, padx=5, pady=5)
    
    # Rainfall column selection
    ttk.Label(frame, text="Rainfall Column:").grid(row=2, column=0, sticky=W, pady=5)
    rainfall_menu = ttk.OptionMenu(frame, rainfall_column, *column_options.get())
    rainfall_menu.grid(row=2, column=1, sticky=W, padx=5, pady=5)
    
    # Output file selection
    ttk.Label(frame, text="Output File:").grid(row=3, column=0, sticky=W, pady=5)
    ttk.Entry(frame, textvariable=output_file, width=50).grid(row=3, column=1, padx=5, pady=5)
    ttk.Button(frame, text="Browse", command=browse_output, bootstyle=PRIMARY).grid(row=3, column=2, padx=5)
    
    # Output format selection
    ttk.Label(frame, text="Output Format:").grid(row=4, column=0, sticky=W, pady=5)
    ttk.OptionMenu(frame, output_format, "xlsx", "csv", "xlsx").grid(row=4, column=1, sticky=W, padx=5, pady=5)
    
    # Data frequency selection
    ttk.Label(frame, text="Data Frequency:").grid(row=5, column=0, sticky=W, pady=5)
    ttk.OptionMenu(frame, data_frequency, "daily", "30-minute", "daily").grid(row=5, column=1, sticky=W, padx=5, pady=5)
    
    # Process button
    ttk.Button(frame, text="Process Rainfall Data", command=process, bootstyle=SUCCESS).grid(row=6, column=0, columnspan=3, pady=20)
    
    # Make the grid columns expandable
    frame.columnconfigure(1, weight=1)
    
    root.mainloop()

if __name__ == "__main__":
    create_gui()