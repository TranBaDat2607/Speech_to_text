"""
EDA - Basic Statistics Analysis for Speech-to-Text Dataset
Phân tích thống kê cơ bản cho dataset speech-to-text tiếng Việt
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_all_segments(data_dir):
    """Load all segments from all topic folders"""
    all_segments = []
    topic_stats = {}
    
    data_path = Path(data_dir)
    
    for topic_folder in data_path.iterdir():
        if topic_folder.is_dir():
            topic_name = topic_folder.name
            segments = []
            
            # Load all JSON files in topic folder
            for json_file in topic_folder.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        segment = json.load(f)
                        segment['topic'] = topic_name
                        segments.append(segment)
                        all_segments.append(segment)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
            
            topic_stats[topic_name] = len(segments)
            print(f"Loaded {len(segments)} segments from {topic_name}")
    
    return all_segments, topic_stats

def analyze_basic_statistics(segments, topic_stats):
    """Analyze basic statistics of the dataset"""
    df = pd.DataFrame(segments)
    
    print("="*60)
    print("BASIC DATASET STATISTICS")
    print("="*60)
    
    # Overall statistics
    print(f"Total topics: {len(topic_stats)}")
    print(f"Total segments: {len(segments)}")
    print(f"Total duration: {df['duration'].sum():.2f} seconds ({df['duration'].sum()/3600:.2f} hours)")
    
    # Duration statistics
    print(f"\nDURATION STATISTICS:")
    print(f"Mean duration: {df['duration'].mean():.2f} seconds")
    print(f"Median duration: {df['duration'].median():.2f} seconds")
    print(f"Min duration: {df['duration'].min():.2f} seconds")
    print(f"Max duration: {df['duration'].max():.2f} seconds")
    print(f"Std duration: {df['duration'].std():.2f} seconds")
    
    # Segments per topic
    print(f"\nSEGMENTS PER TOPIC:")
    topic_df = pd.DataFrame(list(topic_stats.items()), columns=['Topic', 'Segments'])
    topic_df = topic_df.sort_values('Segments', ascending=False)
    print(topic_df.to_string(index=False))
    
    return df, topic_df

def create_visualizations(df, topic_df, output_dir):
    """Create visualization plots"""
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Duration distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['duration'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Segment Durations')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.axvline(df['duration'].mean(), color='red', linestyle='--', label=f'Mean: {df["duration"].mean():.2f}s')
    plt.axvline(df['duration'].median(), color='green', linestyle='--', label=f'Median: {df["duration"].median():.2f}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/duration_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Segments per topic
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(topic_df)), topic_df['Segments'])
    plt.title('Number of Segments per Topic')
    plt.xlabel('Topics')
    plt.ylabel('Number of Segments')
    plt.xticks(range(len(topic_df)), topic_df['Topic'], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/segments_per_topic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Duration by topic (box plot)
    plt.figure(figsize=(15, 8))
    topic_durations = []
    topic_names = []
    
    for topic in topic_df['Topic']:
        topic_data = df[df['topic'] == topic]['duration']
        topic_durations.append(topic_data)
        topic_names.append(topic)
    
    plt.boxplot(topic_durations, labels=topic_names)
    plt.title('Duration Distribution by Topic')
    plt.xlabel('Topics')
    plt.ylabel('Duration (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/duration_by_topic.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cumulative duration by topic
    topic_duration_sum = df.groupby('topic')['duration'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(topic_duration_sum)), topic_duration_sum.values)
    plt.title('Total Duration by Topic')
    plt.xlabel('Topics')
    plt.ylabel('Total Duration (seconds)')
    plt.xticks(range(len(topic_duration_sum)), topic_duration_sum.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/total_duration_by_topic.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_statistics_report(df, topic_df, output_dir):
    """Save detailed statistics to CSV files"""
    
    # Basic statistics
    basic_stats = {
        'Metric': ['Total Topics', 'Total Segments', 'Total Duration (hours)', 
                  'Mean Duration (s)', 'Median Duration (s)', 'Min Duration (s)', 
                  'Max Duration (s)', 'Std Duration (s)'],
        'Value': [len(topic_df), len(df), df['duration'].sum()/3600,
                 df['duration'].mean(), df['duration'].median(), df['duration'].min(),
                 df['duration'].max(), df['duration'].std()]
    }
    
    basic_stats_df = pd.DataFrame(basic_stats)
    basic_stats_df.to_csv(f'{output_dir}/basic_statistics.csv', index=False)
    
    # Topic statistics
    topic_stats_detailed = df.groupby('topic').agg({
        'duration': ['count', 'sum', 'mean', 'median', 'min', 'max', 'std']
    }).round(2)
    
    topic_stats_detailed.columns = ['Segments', 'Total_Duration', 'Mean_Duration', 
                                   'Median_Duration', 'Min_Duration', 'Max_Duration', 'Std_Duration']
    topic_stats_detailed = topic_stats_detailed.sort_values('Total_Duration', ascending=False)
    topic_stats_detailed.to_csv(f'{output_dir}/topic_statistics.csv')
    
    print(f"\nStatistics saved to {output_dir}/")

def main():
    # Configuration
    data_dir = "c:/Users/Admin/Desktop/dat301m/crawl_data/output_segments_grouped"
    output_dir = "c:/Users/Admin/Desktop/dat301m/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    segments, topic_stats = load_all_segments(data_dir)
    
    print("\nAnalyzing basic statistics...")
    df, topic_df = analyze_basic_statistics(segments, topic_stats)
    
    print("\nCreating visualizations...")
    create_visualizations(df, topic_df, output_dir)
    
    print("\nSaving detailed reports...")
    save_statistics_report(df, topic_df, output_dir)
    
    print("\nBasic statistics analysis completed!")

if __name__ == "__main__":
    main()
