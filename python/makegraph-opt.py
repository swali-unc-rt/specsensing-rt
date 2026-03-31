import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pathlib import Path
import argparse

def create_graph_from_csv(csv_filepath):
    # 1. Read and sort the CSV file
    df = pd.read_csv(csv_filepath)
    df = df.sort_values('channelcount')
    
    # Define the metrics we want to plot
    metrics = ['Max', 'P99.9', 'P99', 'P95', 'Average']
    
    # Convert metric values from nanoseconds to microseconds
    for metric in metrics:
        if metric in df.columns:
            df[metric] = df[metric] / 1000.0
            
    # Define specific line styles for each metric
    line_styles = {
        'Max': '-',       
        'P99.9': '-.',    
        'P99': ':',       
        'P95': '--',      
        'Average': '-'    
    }
    
    # Set up the plot size
    plt.figure(figsize=(22, 13))
    ax = plt.gca()
    
    thread_types = df['ThreadType'].unique()
    
    # Map each ThreadType to a distinct color
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color_map = {tt: default_colors[i % len(default_colors)] for i, tt in enumerate(thread_types)}
    
    # List to store label data for the anti-overlap algorithm
    label_data = []
    
    # 2. Plot the lines and collect label coordinates
    for thread_type in thread_types:
        subset = df[df['ThreadType'] == thread_type]
        thread_color = color_map[thread_type]
        
        for metric in metrics:
            style = line_styles[metric]
            
            plt.plot(
                subset['channelcount'], 
                subset[metric], 
                marker='o',
                linestyle=style,
                color=thread_color, 
                linewidth=2,
                label=f'{thread_type} - {metric}'
            )
            
            # Save the raw final coordinates instead of plotting text immediately
            label_data.append({
                'text': f'{thread_type} - {metric}',
                'x': subset['channelcount'].iloc[-1] * 1.02,
                'y': subset[metric].iloc[-1],
                'color': thread_color
            })

    # 3. Anti-Overlap Algorithm for Labels
    # Sort labels by their initial Y position (bottom to top)
    label_data.sort(key=lambda item: item['y'])
    
    # Determine a minimum vertical spacing (e.g., 3% of the total Y-axis data range)
    y_min = df[metrics].min().min()
    y_max = df[metrics].max().max()
    min_y_spacing = (y_max - y_min) * 0.015
    
    # Iterate and push overlapping labels up
    for i in range(1, len(label_data)):
        prev_y = label_data[i-1]['y']
        curr_y = label_data[i]['y']
        
        # If the current label is too close to the one below it, push it up
        if curr_y - prev_y < min_y_spacing:
            label_data[i]['y'] = prev_y + min_y_spacing

    # 4. Render the adjusted labels
    for label in label_data:
        plt.text(
            label['x'], 
            label['y'], 
            label['text'], 
            color=label['color'], 
            fontsize=14, 
            fontweight='bold',
            verticalalignment='center'
        )

    # 5. Format the graph
    plt.xlabel('Channel Count', fontsize=24)
    # Updated label to reflect the conversion
    plt.ylabel('Response Time (µs)', fontsize=24) 
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.title('Response Times by Channel Count and Run Configuration', fontsize=24)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Force the x-axis to only display integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    # Extend the x-axis slightly more to accommodate potentially shifted text
    max_x = df['channelcount'].max()
    plt.xlim(left=df['channelcount'].min(), right=max_x * 1.13)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24)
    #plt.legend().set_visible(False)
    
    plt.tight_layout() 
    
    # 6. Save as PDF
    output_filepath = Path(csv_filepath).with_suffix('.pdf')
    plt.savefig(output_filepath, format='pdf')
    plt.close() 
    
    print(f"Success! Graph saved to: {output_filepath}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot metrics from CSV and save as PDF.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    args = parser.parse_args()
    
    create_graph_from_csv(args.csv_file)