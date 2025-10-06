"""
EDA - Audio Timing Analysis for Speech-to-Text Dataset
Phân tích timing và duration của audio cho dataset speech-to-text tiếng Việt
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import librosa
import warnings
warnings.filterwarnings('ignore')

def find_dataset_folders(datasets_root):
    """Find all dataset folders that contain JSON files"""
    dataset_folders = []
    datasets_path = Path(datasets_root)
    
    for channel_folder in datasets_path.iterdir():
        if channel_folder.is_dir():
            # Check if there's a dataset subfolder
            dataset_subfolder = channel_folder / "dataset"
            if dataset_subfolder.exists() and dataset_subfolder.is_dir():
                # Check if it contains JSON files
                json_files = list(dataset_subfolder.glob("*.json"))
                if json_files:
                    dataset_folders.append({
                        'channel': channel_folder.name,
                        'path': dataset_subfolder,
                        'json_count': len(json_files)
                    })
    
    return dataset_folders

def load_all_segments(datasets_root):
    """Load all segments from all dataset folders"""
    all_segments = []
    
    # Find all dataset folders
    dataset_folders = find_dataset_folders(datasets_root)
    
    if not dataset_folders:
        print(f"No dataset folders found in {datasets_root}")
        return all_segments
    
    print(f"Found {len(dataset_folders)} dataset folders")
    
    for dataset_info in dataset_folders:
        channel_name = dataset_info['channel']
        dataset_path = dataset_info['path']
        
        # Load all JSON files in dataset folder
        for json_file in dataset_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    segment = json.load(f)
                    # Add channel/topic information
                    segment['topic'] = channel_name
                    # Calculate duration if not present
                    if 'duration' not in segment and 'start' in segment and 'end' in segment:
                        segment['duration'] = segment['end'] - segment['start']
                    # Add segment_id for analysis
                    segment['segment_id'] = json_file.stem
                    # Note: audio_path will be set to None since we don't have audio files
                    segment['audio_path'] = None
                    all_segments.append(segment)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return all_segments

def analyze_timing_statistics(segments):
    """Analyze timing and duration statistics"""
    df = pd.DataFrame(segments)
    
    # Calculate additional timing metrics
    df['words_per_second'] = df.apply(lambda row: 
        len(row['text'].split()) / row['duration'] if row['duration'] > 0 else 0, axis=1)
    
    df['chars_per_second'] = df.apply(lambda row: 
        len(row['text']) / row['duration'] if row['duration'] > 0 else 0, axis=1)
    
    print("="*60)
    print("AUDIO TIMING ANALYSIS")
    print("="*60)
    
    print(f"DURATION STATISTICS:")
    print(f"Mean duration: {df['duration'].mean():.2f} seconds")
    print(f"Median duration: {df['duration'].median():.2f} seconds")
    print(f"Min duration: {df['duration'].min():.2f} seconds")
    print(f"Max duration: {df['duration'].max():.2f} seconds")
    print(f"Standard deviation: {df['duration'].std():.2f} seconds")
    print(f"25th percentile: {df['duration'].quantile(0.25):.2f} seconds")
    print(f"75th percentile: {df['duration'].quantile(0.75):.2f} seconds")
    
    print(f"\nSPEAKING RATE ANALYSIS:")
    print(f"Mean words per second: {df['words_per_second'].mean():.2f}")
    print(f"Median words per second: {df['words_per_second'].median():.2f}")
    print(f"Min words per second: {df['words_per_second'].min():.2f}")
    print(f"Max words per second: {df['words_per_second'].max():.2f}")
    
    print(f"\nCHARACTER RATE ANALYSIS:")
    print(f"Mean chars per second: {df['chars_per_second'].mean():.2f}")
    print(f"Median chars per second: {df['chars_per_second'].median():.2f}")
    
    # Duration categories
    df['duration_category'] = pd.cut(df['duration'], 
                                   bins=[0, 10, 20, 30, 60, float('inf')],
                                   labels=['Very Short (0-10s)', 'Short (10-20s)', 
                                          'Medium (20-30s)', 'Long (30-60s)', 'Very Long (60s+)'])
    
    print(f"\nDURATION CATEGORIES:")
    duration_counts = df['duration_category'].value_counts()
    for category, count in duration_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{category}: {count} segments ({percentage:.1f}%)")
    
    return df

def analyze_timing_by_topic(df):
    """Analyze timing statistics by topic"""
    print(f"\nTIMING STATISTICS BY TOPIC:")
    
    topic_timing = df.groupby('topic').agg({
        'duration': ['count', 'mean', 'median', 'min', 'max', 'sum'],
        'words_per_second': ['mean', 'median'],
        'chars_per_second': ['mean', 'median']
    }).round(2)
    
    topic_timing.columns = ['Segments', 'Mean_Duration', 'Median_Duration', 
                           'Min_Duration', 'Max_Duration', 'Total_Duration',
                           'Mean_WPS', 'Median_WPS', 'Mean_CPS', 'Median_CPS']
    
    topic_timing = topic_timing.sort_values('Total_Duration', ascending=False)
    print(topic_timing.head(10).to_string())
    
    return topic_timing

def check_audio_files_existence(df, sample_size=100):
    """Check if audio files exist and get basic audio properties"""
    print(f"\nAUDIO FILES VALIDATION:")
    
    # Sample some files to check
    sample_df = df.sample(min(sample_size, len(df)))
    
    existing_files = 0
    audio_properties = []
    
    for idx, row in sample_df.iterrows():
        audio_path = row['audio_path']
        
        if os.path.exists(audio_path):
            existing_files += 1
            
            try:
                # Load audio file to get properties
                y, sr = librosa.load(audio_path, sr=None)
                actual_duration = len(y) / sr
                
                audio_properties.append({
                    'file': os.path.basename(audio_path),
                    'expected_duration': row['duration'],
                    'actual_duration': actual_duration,
                    'duration_diff': abs(actual_duration - row['duration']),
                    'sample_rate': sr,
                    'samples': len(y)
                })
                
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
    
    print(f"Audio files checked: {len(sample_df)}")
    print(f"Existing files: {existing_files} ({existing_files/len(sample_df)*100:.1f}%)")
    
    if audio_properties:
        props_df = pd.DataFrame(audio_properties)
        print(f"\nAUDIO PROPERTIES (sample of {len(props_df)} files):")
        print(f"Mean sample rate: {props_df['sample_rate'].mean():.0f} Hz")
        print(f"Sample rate range: {props_df['sample_rate'].min():.0f} - {props_df['sample_rate'].max():.0f} Hz")
        print(f"Mean duration difference: {props_df['duration_diff'].mean():.3f} seconds")
        print(f"Max duration difference: {props_df['duration_diff'].max():.3f} seconds")
        
        return props_df
    
    return None

def detect_timing_anomalies(df):
    """Detect potential timing anomalies"""
    print(f"\nTIMING ANOMALIES DETECTION:")
    
    anomalies = []
    
    # Very short segments (< 1 second)
    very_short = df[df['duration'] < 1.0]
    if len(very_short) > 0:
        anomalies.append(f"Very short segments (< 1s): {len(very_short)} segments")
    
    # Very long segments (> 120 seconds)
    very_long = df[df['duration'] > 120.0]
    if len(very_long) > 0:
        anomalies.append(f"Very long segments (> 2min): {len(very_long)} segments")
    
    # Segments with zero duration
    zero_duration = df[df['duration'] == 0]
    if len(zero_duration) > 0:
        anomalies.append(f"Zero duration segments: {len(zero_duration)} segments")
    
    # Segments with very high speaking rate (> 8 words/second)
    fast_speech = df[df['words_per_second'] > 8.0]
    if len(fast_speech) > 0:
        anomalies.append(f"Very fast speech (> 8 words/s): {len(fast_speech)} segments")
    
    # Segments with very low speaking rate (< 0.5 words/second)
    slow_speech = df[df['words_per_second'] < 0.5]
    if len(slow_speech) > 0:
        anomalies.append(f"Very slow speech (< 0.5 words/s): {len(slow_speech)} segments")
    
    if anomalies:
        for anomaly in anomalies:
            print(f"- {anomaly}")
    else:
        print("No significant timing anomalies detected.")
    
    return anomalies

def create_timing_visualizations(df, output_dir):
    """Create timing analysis visualizations"""
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Duration distribution with different scales
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Linear scale
    axes[0,0].hist(df['duration'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Duration Distribution (Linear Scale)')
    axes[0,0].set_xlabel('Duration (seconds)')
    axes[0,0].set_ylabel('Frequency')
    axes[0,0].axvline(df['duration'].mean(), color='red', linestyle='--', label='Mean')
    axes[0,0].axvline(df['duration'].median(), color='green', linestyle='--', label='Median')
    axes[0,0].legend()
    
    # Log scale
    axes[0,1].hist(df['duration'], bins=50, alpha=0.7, edgecolor='black')
    axes[0,1].set_yscale('log')
    axes[0,1].set_title('Duration Distribution (Log Scale)')
    axes[0,1].set_xlabel('Duration (seconds)')
    axes[0,1].set_ylabel('Frequency (log)')
    
    # Box plot
    axes[1,0].boxplot(df['duration'])
    axes[1,0].set_title('Duration Box Plot')
    axes[1,0].set_ylabel('Duration (seconds)')
    
    # Duration categories
    duration_counts = df['duration_category'].value_counts()
    axes[1,1].pie(duration_counts.values, labels=duration_counts.index, autopct='%1.1f%%')
    axes[1,1].set_title('Duration Categories Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/duration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Speaking rate analysis
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Words per second distribution
    wps_filtered = df['words_per_second'][(df['words_per_second'] > 0) & (df['words_per_second'] < 10)]
    axes[0].hist(wps_filtered, bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_title('Words per Second Distribution')
    axes[0].set_xlabel('Words per Second')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(wps_filtered.mean(), color='red', linestyle='--', label='Mean')
    axes[0].axvline(wps_filtered.median(), color='green', linestyle='--', label='Median')
    axes[0].legend()
    
    # Characters per second distribution
    cps_filtered = df['chars_per_second'][(df['chars_per_second'] > 0) & (df['chars_per_second'] < 50)]
    axes[1].hist(cps_filtered, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_title('Characters per Second Distribution')
    axes[1].set_xlabel('Characters per Second')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(cps_filtered.mean(), color='red', linestyle='--', label='Mean')
    axes[1].axvline(cps_filtered.median(), color='green', linestyle='--', label='Median')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speaking_rate_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Duration by topic
    plt.figure(figsize=(15, 8))
    topic_durations = []
    topic_names = []
    
    for topic in df['topic'].unique():
        topic_data = df[df['topic'] == topic]['duration']
        topic_durations.append(topic_data)
        topic_names.append(topic)
    
    plt.boxplot(topic_durations, labels=topic_names)
    plt.title('Duration Distribution by Topic')
    plt.xlabel('Topics')
    plt.ylabel('Duration (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/duration_by_topic_boxplot.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Speaking rate vs Duration scatter plot
    plt.figure(figsize=fig_size)
    plt.scatter(df['duration'], df['words_per_second'], alpha=0.6)
    plt.title('Speaking Rate vs Duration')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Words per Second')
    
    # Add correlation coefficient
    correlation = df['duration'].corr(df['words_per_second'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speaking_rate_vs_duration.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_timing_analysis_report(df, topic_timing, anomalies, audio_props, output_dir):
    """Save timing analysis results to files"""
    
    # Timing statistics summary
    timing_stats = {
        'Metric': ['Mean Duration', 'Median Duration', 'Min Duration', 'Max Duration',
                  'Std Duration', 'Total Duration (hours)', 'Mean Words/Second', 
                  'Median Words/Second', 'Mean Chars/Second'],
        'Value': [df['duration'].mean(), df['duration'].median(), df['duration'].min(),
                 df['duration'].max(), df['duration'].std(), df['duration'].sum()/3600,
                 df['words_per_second'].mean(), df['words_per_second'].median(),
                 df['chars_per_second'].mean()]
    }
    
    timing_stats_df = pd.DataFrame(timing_stats)
    timing_stats_df.to_csv(f'{output_dir}/timing_statistics.csv', index=False)
    
    # Topic timing statistics
    topic_timing.to_csv(f'{output_dir}/topic_timing_statistics.csv')
    
    # Duration categories
    duration_categories = df['duration_category'].value_counts().reset_index()
    duration_categories.columns = ['Category', 'Count']
    duration_categories['Percentage'] = (duration_categories['Count'] / len(df)) * 100
    duration_categories.to_csv(f'{output_dir}/duration_categories.csv', index=False)
    
    # Anomalies report
    with open(f'{output_dir}/timing_anomalies.txt', 'w', encoding='utf-8') as f:
        f.write("TIMING ANOMALIES REPORT\n")
        f.write("="*50 + "\n\n")
        if anomalies:
            for anomaly in anomalies:
                f.write(f"- {anomaly}\n")
        else:
            f.write("No significant timing anomalies detected.\n")
    
    # Audio properties (if available)
    if audio_props is not None:
        audio_props.to_csv(f'{output_dir}/audio_properties_sample.csv', index=False)
    
    # Detailed segment timing analysis
    segment_timing = df[['topic', 'segment_id', 'duration', 'words_per_second', 
                        'chars_per_second', 'duration_category']].copy()
    segment_timing.to_csv(f'{output_dir}/segment_timing_analysis.csv', index=False)
    
    print(f"\nTiming analysis results saved to {output_dir}/")

def main():
    # Configuration
    datasets_root = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/crawl_data/datasets"
    output_dir = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scanning for datasets...")
    segments = load_all_segments(datasets_root)
    
    if not segments:
        print("No segments found! Please check the datasets directory.")
        return
    
    print(f"Total segments loaded: {len(segments)}")
    print("Analyzing timing statistics...")
    df = analyze_timing_statistics(segments)
    
    print("Analyzing timing by topic...")
    topic_timing = analyze_timing_by_topic(df)
    
    print("Checking audio files...")
    print("Note: Audio files validation skipped as we only have JSON metadata")
    audio_props = None
    
    print("Detecting timing anomalies...")
    anomalies = detect_timing_anomalies(df)
    
    print("\nCreating timing visualizations...")
    create_timing_visualizations(df, output_dir)
    
    print("Saving timing analysis reports...")
    save_timing_analysis_report(df, topic_timing, anomalies, audio_props, output_dir)
    
    print("\nAudio timing analysis completed!")

if __name__ == "__main__":
    main()
