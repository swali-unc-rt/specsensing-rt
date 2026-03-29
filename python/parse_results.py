import argparse
import sys
import pandas as pd

def calculate_statistics(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """
    Groups the dataframe by the specified column and calculates the 
    max, 99.9th, 99th, 95th percentiles, and average for the 'response' column.
    """
    # Using pandas named aggregation for clean output columns
    stats = df.groupby(group_column).agg(
        Max=('Response', 'max'),
        P99_9=('Response', lambda x: x.quantile(0.999)),
        P99=('Response', lambda x: x.quantile(0.99)),
        P95=('Response', lambda x: x.quantile(0.95)),
        Average=('Response', 'mean')
    ).reset_index()
    
    return stats

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strips whitespace from column names and all string values."""
    # Strip spaces from column headers
    df.columns = df.columns.str.strip()
    
    # Strip spaces from string (object) columns
    for col in df.select_dtypes(['object']).columns:
        df[col] = df[col].str.strip()
        
    return df

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Calculate response time statistics merged from two CSV files."
    )
    parser.add_argument("csv1", help="Path to the first CSV (thread definitions)")
    parser.add_argument("csv2", help="Path to the second CSV (job metrics)")
    
    args = parser.parse_args()

    csv2_columns = ['# Task', 'Job', 'Period', 'Response', 'DL Miss?', 'Lateness', 'Tardiness', 'Forced?', 'ACET', 'Preemptions', 'Migrations']

    # Load the CSV files
    try:
        df1 = pd.read_csv(args.csv1, skipinitialspace=True)
        df2 = pd.read_csv(args.csv2, comment='#', skipinitialspace=True, header=None, names=csv2_columns)
        df2 = clean_dataframe(df2)
    except FileNotFoundError as e:
        print(f"Error: Could not find file. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the CSVs: {e}")
        sys.exit(1)

    # Ensure 'threadid' exists in both dataframes before merging
    if 'threadid' not in df1.columns or '# Task' not in df2.columns:
        print("Error: 'threadid' column must exist in the first CSV and '# Task' column must exist in the second CSV.")
        print(f"Columns in CSV1: {list(df1.columns)}")
        print(f"Columns in CSV2: {list(df2.columns)}")
        sys.exit(1)
    
    # print(df1['threadid'].dtype)
    # print(df2['# Task'].dtype)

    # Merge the two dataframes on 'threadid' and '# Task'
    # using an inner join so we only calculate stats for matching threads
    merged_df = pd.merge(df1, df2, left_on='threadid', right_on='# Task', how='inner')
    
    # Ensure 'response' is treated as a numeric column
    merged_df['Response'] = pd.to_numeric(merged_df['Response'], errors='coerce')

    # Calculate and display stats for ThreadType
    if 'ThreadType' in merged_df.columns:
        print("\n--- Statistics by ThreadType ---")
        threadtype_stats = calculate_statistics(merged_df, 'ThreadType')
        print(threadtype_stats.to_string(index=False))
    else:
        print("\nWarning: 'ThreadType' column not found in the first CSV.")

    # Calculate and display stats for dagid
    if 'dagid' in merged_df.columns:
        print("\n--- Statistics by dagid ---")
        dagid_stats = calculate_statistics(merged_df, 'dagid')
        print(dagid_stats.to_string(index=False))
    else:
        print("\nWarning: 'dagid' column not found in the first CSV.")

if __name__ == "__main__":
    main()