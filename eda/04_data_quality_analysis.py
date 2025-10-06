"""
EDA - Data Quality Analysis for Speech-to-Text Dataset
Phân tích chất lượng dữ liệu cho dataset speech-to-text tiếng Việt
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re

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
    loading_errors = []
    
    # Find all dataset folders
    dataset_folders = find_dataset_folders(datasets_root)
    
    if not dataset_folders:
        print(f"No dataset folders found in {datasets_root}")
        return all_segments, loading_errors
    
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
                    segment['file_path'] = str(json_file)
                    all_segments.append(segment)
            except Exception as e:
                loading_errors.append({
                    'file': str(json_file),
                    'error': str(e)
                })
    
    return all_segments, loading_errors

def check_data_completeness(segments):
    """Check for missing or incomplete data"""
    df = pd.DataFrame(segments)
    
    print("="*60)
    print("DATA COMPLETENESS ANALYSIS")
    print("="*60)
    
    required_fields = ['segment_id', 'start', 'end', 'duration', 'text', 'video_id']
    
    print(f"Total segments loaded: {len(segments)}")
    print(f"\nMISSING VALUES BY FIELD:")
    
    missing_data = {}
    for field in required_fields:
        if field in df.columns:
            missing_count = df[field].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            missing_data[field] = {'count': missing_count, 'percentage': missing_percentage}
            print(f"{field}: {missing_count} missing ({missing_percentage:.2f}%)")
        else:
            missing_data[field] = {'count': len(df), 'percentage': 100.0}
            print(f"{field}: FIELD NOT FOUND (100.00%)")
    
    return df, missing_data

def check_data_consistency(df):
    """Check for data consistency issues"""
    print(f"\nDATA CONSISTENCY CHECKS:")
    
    consistency_issues = []
    
    # 1. Check if start <= end
    if 'start' in df.columns and 'end' in df.columns:
        invalid_time_order = df[df['start'] > df['end']]
        if len(invalid_time_order) > 0:
            issue = f"Invalid time order (start > end): {len(invalid_time_order)} segments"
            consistency_issues.append(issue)
            print(f"- {issue}")
    
    # 2. Check if duration matches (end - start)
    if all(col in df.columns for col in ['start', 'end', 'duration']):
        df['calculated_duration'] = df['end'] - df['start']
        duration_mismatch = df[abs(df['duration'] - df['calculated_duration']) > 0.1]
        if len(duration_mismatch) > 0:
            issue = f"Duration mismatch: {len(duration_mismatch)} segments"
            consistency_issues.append(issue)
            print(f"- {issue}")
    
    # 3. Check for negative durations
    if 'duration' in df.columns:
        negative_duration = df[df['duration'] < 0]
        if len(negative_duration) > 0:
            issue = f"Negative duration: {len(negative_duration)} segments"
            consistency_issues.append(issue)
            print(f"- {issue}")
    
    # 4. Check for zero duration
    if 'duration' in df.columns:
        zero_duration = df[df['duration'] == 0]
        if len(zero_duration) > 0:
            issue = f"Zero duration: {len(zero_duration)} segments"
            consistency_issues.append(issue)
            print(f"- {issue}")
    
    # 5. Check segment_id consistency within topics
    if 'segment_id' in df.columns:
        segment_id_issues = []
        for topic in df['topic'].unique():
            topic_data = df[df['topic'] == topic].sort_values('segment_id')
            expected_ids = list(range(len(topic_data)))
            actual_ids = sorted(topic_data['segment_id'].tolist())
            
            if actual_ids != expected_ids:
                segment_id_issues.append(topic)
        
        if segment_id_issues:
            issue = f"Segment ID inconsistency in {len(segment_id_issues)} topics"
            consistency_issues.append(issue)
            print(f"- {issue}")
    
    if not consistency_issues:
        print("- No major consistency issues found")
    
    return consistency_issues

def check_text_quality(df):
    """Check text quality issues"""
    print(f"\nTEXT QUALITY ANALYSIS:")
    
    text_issues = []
    
    if 'text' not in df.columns:
        print("- Text field not found")
        return text_issues
    
    # 1. Empty or very short text
    empty_text = df[df['text'].isnull() | (df['text'].str.len() == 0)]
    if len(empty_text) > 0:
        issue = f"Empty text: {len(empty_text)} segments"
        text_issues.append(issue)
        print(f"- {issue}")
    
    very_short_text = df[df['text'].str.len() < 5]
    if len(very_short_text) > 0:
        issue = f"Very short text (< 5 chars): {len(very_short_text)} segments"
        text_issues.append(issue)
        print(f"- {issue}")
    
    # 2. Text with only special characters or numbers
    df['text_clean'] = df['text'].fillna('').str.replace(r'[^\w\s\u00C0-\u1EF9]', '', regex=True)
    only_special_chars = df[df['text_clean'].str.len() == 0]
    if len(only_special_chars) > 0:
        issue = f"Text with only special characters: {len(only_special_chars)} segments"
        text_issues.append(issue)
        print(f"- {issue}")
    
    # 3. Extremely long text (potential transcription errors)
    very_long_text = df[df['text'].str.len() > 1000]
    if len(very_long_text) > 0:
        issue = f"Very long text (> 1000 chars): {len(very_long_text)} segments"
        text_issues.append(issue)
        print(f"- {issue}")
    
    # 4. Text with unusual character patterns
    unusual_patterns = []
    
    # Repeated characters (e.g., "aaaaaaa") - suppress warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        repeated_chars = df[df['text'].str.contains(r'(.)\1{5,}', na=False, regex=True)]
    if len(repeated_chars) > 0:
        unusual_patterns.append(f"Repeated characters: {len(repeated_chars)} segments")
    
    # Too many numbers
    many_numbers = df[df['text'].str.count(r'\d') > 20]
    if len(many_numbers) > 0:
        unusual_patterns.append(f"Text with many numbers: {len(many_numbers)} segments")
    
    # Too many special characters
    many_special = df[df['text'].str.count(r'[^\w\s\u00C0-\u1EF9]') > 50]
    if len(many_special) > 0:
        unusual_patterns.append(f"Text with many special chars: {len(many_special)} segments")
    
    if unusual_patterns:
        for pattern in unusual_patterns:
            text_issues.append(pattern)
            print(f"- {pattern}")
    
    # 5. Text quality metrics
    df['text'] = df['text'].fillna('')  # Handle NaN values
    df['word_count'] = df['text'].str.split().str.len()
    df['char_count'] = df['text'].str.len()
    df['avg_word_length'] = df.apply(lambda row: 
        sum(len(word) for word in str(row['text']).split()) / len(str(row['text']).split()) 
        if str(row['text']).strip() and len(str(row['text']).split()) > 0 else 0, axis=1)
    
    print(f"\nTEXT QUALITY METRICS:")
    print(f"Mean words per segment: {df['word_count'].mean():.2f}")
    print(f"Mean characters per segment: {df['char_count'].mean():.2f}")
    print(f"Mean word length: {df['avg_word_length'].mean():.2f} characters")
    
    return text_issues

def check_video_id_references(df):
    """Check video ID references"""
    print(f"\nVIDEO ID REFERENCE ANALYSIS:")
    
    video_issues = []
    
    if 'video_id' not in df.columns:
        print("- Video ID field not found")
        return video_issues
    
    # 1. Missing video ID references
    missing_video_ref = df[df['video_id'].isnull() | (df['video_id'].str.len() == 0)]
    if len(missing_video_ref) > 0:
        issue = f"Missing video ID reference: {len(missing_video_ref)} segments"
        video_issues.append(issue)
        print(f"- {issue}")
    
    # 2. Check video ID format consistency (YouTube IDs are typically 11 characters)
    if 'video_id' in df.columns:
        invalid_length = df[df['video_id'].str.len() != 11]
        if len(invalid_length) > 0:
            issue = f"Invalid video ID length: {len(invalid_length)} segments"
            video_issues.append(issue)
            print(f"- {issue}")
        
        # Check for invalid characters in video IDs
        invalid_chars = df[~df['video_id'].str.match(r'^[A-Za-z0-9_-]+$', na=False)]
        if len(invalid_chars) > 0:
            issue = f"Invalid characters in video ID: {len(invalid_chars)} segments"
            video_issues.append(issue)
            print(f"- {issue}")
    
    # 3. Check video ID distribution
    video_id_counts = df['video_id'].value_counts()
    print(f"- Total unique video IDs: {len(video_id_counts)}")
    print(f"- Average segments per video: {video_id_counts.mean():.2f}")
    
    # Videos with very few segments (might indicate incomplete processing)
    few_segments = video_id_counts[video_id_counts < 3]
    if len(few_segments) > 0:
        issue = f"Videos with very few segments (< 3): {len(few_segments)} videos"
        video_issues.append(issue)
        print(f"- {issue}")
    
    return video_issues

def analyze_data_distribution(df):
    """Analyze data distribution across topics"""
    print(f"\nDATA DISTRIBUTION ANALYSIS:")
    
    topic_stats = df.groupby('topic').agg({
        'segment_id': 'count',
        'duration': ['sum', 'mean'],
        'text': lambda x: x.str.len().mean()
    }).round(2)
    
    topic_stats.columns = ['Segments', 'Total_Duration', 'Mean_Duration', 'Mean_Text_Length']
    topic_stats = topic_stats.sort_values('Segments', ascending=False)
    
    print(f"Data distribution by topic (top 10):")
    print(topic_stats.head(10).to_string())
    
    # Check for imbalanced data
    min_segments = topic_stats['Segments'].min()
    max_segments = topic_stats['Segments'].max()
    imbalance_ratio = max_segments / min_segments if min_segments > 0 else float('inf')
    
    print(f"\nData balance analysis:")
    print(f"Min segments per topic: {min_segments}")
    print(f"Max segments per topic: {max_segments}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 5:
        print("WARNING: Significant data imbalance detected")
    
    return topic_stats

def create_quality_visualizations(df, output_dir):
    """Create data quality visualizations"""
    plt.style.use('default')
    
    # 1. Missing data heatmap
    if len(df.columns) > 0:
        plt.figure(figsize=(10, 6))
        missing_data = df.isnull()
        if missing_data.any().any():
            sns.heatmap(missing_data, cbar=True, yticklabels=False, cmap='viridis')
            plt.title('Missing Data Pattern')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/missing_data_heatmap.png', dpi=300, bbox_inches='tight')
            plt.show()
        else:
            print("No missing data to visualize")
    
    # 2. Text length distribution
    if 'text' in df.columns:
        plt.figure(figsize=(12, 8))
        text_lengths = df['text'].fillna('').str.len()
        
        plt.subplot(2, 2, 1)
        plt.hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Text Length Distribution')
        plt.xlabel('Characters')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 2)
        plt.boxplot(text_lengths)
        plt.title('Text Length Box Plot')
        plt.ylabel('Characters')
        
        plt.subplot(2, 2, 3)
        word_counts = df['text'].fillna('').str.split().str.len()
        plt.hist(word_counts, bins=50, alpha=0.7, edgecolor='black')
        plt.title('Word Count Distribution')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        
        plt.subplot(2, 2, 4)
        plt.boxplot(word_counts)
        plt.title('Word Count Box Plot')
        plt.ylabel('Words')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/text_quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Data distribution by topic
    topic_counts = df['topic'].value_counts()
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(topic_counts)), topic_counts.values)
    plt.title('Segments per Topic')
    plt.xlabel('Topics')
    plt.ylabel('Number of Segments')
    plt.xticks(range(len(topic_counts)), topic_counts.index, rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/data_distribution_by_topic.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_quality_analysis_report(df, missing_data, consistency_issues, text_issues, 
                               video_issues, topic_stats, loading_errors, output_dir):
    """Save data quality analysis results"""
    
    # Quality summary report
    quality_summary = {
        'Category': ['Data Loading', 'Missing Data', 'Consistency', 'Text Quality', 'Video References'],
        'Issues_Count': [len(loading_errors), sum(missing_data[field]['count'] for field in missing_data),
                        len(consistency_issues), len(text_issues), len(video_issues)],
        'Status': ['OK' if len(loading_errors) == 0 else 'ISSUES',
                  'OK' if sum(missing_data[field]['count'] for field in missing_data) == 0 else 'ISSUES',
                  'OK' if len(consistency_issues) == 0 else 'ISSUES',
                  'OK' if len(text_issues) == 0 else 'ISSUES',
                  'OK' if len(video_issues) == 0 else 'ISSUES']
    }
    
    quality_summary_df = pd.DataFrame(quality_summary)
    quality_summary_df.to_csv(f'{output_dir}/data_quality_summary.csv', index=False)
    
    # Detailed missing data report
    missing_data_df = pd.DataFrame([
        {'Field': field, 'Missing_Count': data['count'], 'Missing_Percentage': data['percentage']}
        for field, data in missing_data.items()
    ])
    missing_data_df.to_csv(f'{output_dir}/missing_data_report.csv', index=False)
    
    # Topic statistics
    topic_stats.to_csv(f'{output_dir}/topic_data_distribution.csv')
    
    # Detailed quality issues report
    with open(f'{output_dir}/data_quality_issues.txt', 'w', encoding='utf-8') as f:
        f.write("DATA QUALITY ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("LOADING ERRORS:\n")
        if loading_errors:
            for error in loading_errors:
                f.write(f"- {error['file']}: {error['error']}\n")
        else:
            f.write("- No loading errors\n")
        
        f.write("\nCONSISTENCY ISSUES:\n")
        if consistency_issues:
            for issue in consistency_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("- No consistency issues\n")
        
        f.write("\nTEXT QUALITY ISSUES:\n")
        if text_issues:
            for issue in text_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("- No text quality issues\n")
        
        f.write("\nVIDEO REFERENCE ISSUES:\n")
        if video_issues:
            for issue in video_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("- No video reference issues\n")
    
    print(f"\nData quality analysis results saved to {output_dir}/")

def main():
    # Configuration
    datasets_root = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/crawl_data/datasets"
    output_dir = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scanning for datasets and checking for loading errors...")
    segments, loading_errors = load_all_segments(datasets_root)
    
    if not segments:
        print("No segments found! Please check the datasets directory.")
        return
    
    if loading_errors:
        print(f"Found {len(loading_errors)} loading errors")
        for error in loading_errors[:5]:  # Show first 5 errors
            print(f"- {error['file']}: {error['error']}")
    
    print(f"Total segments loaded: {len(segments)}")
    print("Checking data completeness...")
    df, missing_data = check_data_completeness(segments)
    
    print("Checking data consistency...")
    consistency_issues = check_data_consistency(df)
    
    print("Analyzing text quality...")
    text_issues = check_text_quality(df)
    
    print("Checking video ID references...")
    video_issues = check_video_id_references(df)
    
    print("Analyzing data distribution...")
    topic_stats = analyze_data_distribution(df)
    
    print("\nCreating quality visualizations...")
    create_quality_visualizations(df, output_dir)
    
    print("Saving quality analysis reports...")
    save_quality_analysis_report(df, missing_data, consistency_issues, text_issues, 
                                video_issues, topic_stats, loading_errors, output_dir)
    
    print("\nData quality analysis completed!")
    
    # Summary
    total_issues = (len(loading_errors) + len(consistency_issues) + 
                   len(text_issues) + len(video_issues))
    
    if total_issues == 0:
        print("✓ No major data quality issues found!")
    else:
        print(f"⚠ Found {total_issues} potential data quality issues")
        print("Check the detailed reports in the outputs folder for more information.")

if __name__ == "__main__":
    main()
