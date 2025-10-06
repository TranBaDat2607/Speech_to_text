"""
EDA - Text Analysis for Speech-to-Text Dataset
Phân tích văn bản cho dataset speech-to-text tiếng Việt
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
import re
from wordcloud import WordCloud

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
                    all_segments.append(segment)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return all_segments

def preprocess_text(text):
    """Basic text preprocessing for Vietnamese"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters but keep Vietnamese characters
    text = re.sub(r'[^\w\s\u00C0-\u1EF9]', '', text)
    
    return text.lower()

def analyze_text_statistics(segments):
    """Analyze text-related statistics"""
    df = pd.DataFrame(segments)
    
    # Text preprocessing - handle NaN values
    df['text'] = df['text'].fillna('')
    df['text_clean'] = df['text'].apply(preprocess_text)
    
    # Text quality metrics
    df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()) if x else 0)
    df['char_count'] = df['text_clean'].apply(len)
    df['sentence_count'] = df['text'].apply(lambda x: len(re.split(r'[.!?]+', str(x))) if x else 0)
    
    # Speaking rate (words per second)
    df['speaking_rate'] = df['word_count'] / df['duration']
    df['speaking_rate'] = df['speaking_rate'].replace([np.inf, -np.inf], 0)
    
    print("="*60)
    print("TEXT ANALYSIS STATISTICS")
    print("="*60)
    
    print(f"WORD COUNT STATISTICS:")
    print(f"Mean words per segment: {df['word_count'].mean():.2f}")
    print(f"Median words per segment: {df['word_count'].median():.2f}")
    print(f"Min words per segment: {df['word_count'].min()}")
    print(f"Max words per segment: {df['word_count'].max()}")
    print(f"Total words in dataset: {df['word_count'].sum():,}")
    
    print(f"\nCHARACTER COUNT STATISTICS:")
    print(f"Mean characters per segment: {df['char_count'].mean():.2f}")
    print(f"Median characters per segment: {df['char_count'].median():.2f}")
    print(f"Min characters per segment: {df['char_count'].min()}")
    print(f"Max characters per segment: {df['char_count'].max()}")
    
    print(f"\nSPEAKING RATE STATISTICS:")
    print(f"Mean speaking rate: {df['speaking_rate'].mean():.2f} words/second")
    print(f"Median speaking rate: {df['speaking_rate'].median():.2f} words/second")
    print(f"Min speaking rate: {df['speaking_rate'].min():.2f} words/second")
    print(f"Max speaking rate: {df['speaking_rate'].max():.2f} words/second")
    
    return df

def analyze_vocabulary(df):
    """Analyze vocabulary and word frequency"""
    print(f"\nVOCABULARY ANALYSIS:")
    
    # Combine all text
    all_text = ' '.join(df['text_clean'].fillna(''))
    words = all_text.split()
    
    # Word frequency
    word_freq = Counter(words)
    unique_words = len(word_freq)
    total_words = len(words)
    
    print(f"Total unique words: {unique_words:,}")
    print(f"Total words: {total_words:,}")
    print(f"Vocabulary richness: {unique_words/total_words:.4f}")
    
    # Most common words
    print(f"\nTOP 20 MOST COMMON WORDS:")
    for word, count in word_freq.most_common(20):
        print(f"{word}: {count}")
    
    # Words appearing only once
    hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
    print(f"\nWords appearing only once: {hapax_legomena} ({hapax_legomena/unique_words*100:.2f}%)")
    
    return word_freq, unique_words, total_words

def analyze_text_by_topic(df):
    """Analyze text statistics by topic"""
    print(f"\nTEXT STATISTICS BY TOPIC:")
    
    topic_stats = df.groupby('topic').agg({
        'word_count': ['mean', 'median', 'sum'],
        'char_count': ['mean', 'median', 'sum'],
        'speaking_rate': ['mean', 'median'],
        'duration': 'sum'
    }).round(2)
    
    topic_stats.columns = ['Mean_Words', 'Median_Words', 'Total_Words',
                          'Mean_Chars', 'Median_Chars', 'Total_Chars',
                          'Mean_Speaking_Rate', 'Median_Speaking_Rate', 'Total_Duration']
    
    topic_stats = topic_stats.sort_values('Total_Words', ascending=False)
    print(topic_stats.head(10).to_string())
    
    return topic_stats

def create_text_visualizations(df, word_freq, output_dir):
    """Create text analysis visualizations"""
    plt.style.use('default')
    fig_size = (12, 8)
    
    # 1. Word count distribution
    plt.figure(figsize=fig_size)
    plt.hist(df['word_count'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Word Count per Segment')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.axvline(df['word_count'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["word_count"].mean():.1f}')
    plt.axvline(df['word_count'].median(), color='green', linestyle='--', 
                label=f'Median: {df["word_count"].median():.1f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Speaking rate distribution
    plt.figure(figsize=fig_size)
    # Filter out extreme values for better visualization
    speaking_rate_filtered = df['speaking_rate'][(df['speaking_rate'] > 0) & (df['speaking_rate'] < 10)]
    plt.hist(speaking_rate_filtered, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Distribution of Speaking Rate (Words per Second)')
    plt.xlabel('Speaking Rate (words/second)')
    plt.ylabel('Frequency')
    plt.axvline(speaking_rate_filtered.mean(), color='red', linestyle='--', 
                label=f'Mean: {speaking_rate_filtered.mean():.2f}')
    plt.axvline(speaking_rate_filtered.median(), color='green', linestyle='--', 
                label=f'Median: {speaking_rate_filtered.median():.2f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/speaking_rate_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Word count vs Duration scatter plot
    plt.figure(figsize=fig_size)
    plt.scatter(df['duration'], df['word_count'], alpha=0.6)
    plt.title('Word Count vs Duration')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Word Count')
    
    # Add correlation coefficient
    correlation = df['duration'].corr(df['word_count'])
    plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/word_count_vs_duration.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Top 20 most frequent words
    top_words = dict(word_freq.most_common(20))
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.title('Top 20 Most Frequent Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_words_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Word cloud
    try:
        all_text = ' '.join(df['text_clean'].fillna(''))
        wordcloud = WordCloud(width=1200, height=600, 
                             background_color='white',
                             max_words=100,
                             font_path=None,  # You might need to specify Vietnamese font
                             colormap='viridis').generate(all_text)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Most Frequent Words')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Could not generate word cloud: {e}")

def save_text_analysis_report(df, word_freq, topic_stats, output_dir):
    """Save text analysis results to files"""
    
    # Text statistics summary
    text_stats = {
        'Metric': ['Mean Words per Segment', 'Median Words per Segment', 
                  'Total Words', 'Unique Words', 'Vocabulary Richness',
                  'Mean Speaking Rate', 'Median Speaking Rate'],
        'Value': [df['word_count'].mean(), df['word_count'].median(),
                 df['word_count'].sum(), len(word_freq), 
                 len(word_freq)/df['word_count'].sum(),
                 df['speaking_rate'].mean(), df['speaking_rate'].median()]
    }
    
    text_stats_df = pd.DataFrame(text_stats)
    text_stats_df.to_csv(f'{output_dir}/text_statistics.csv', index=False)
    
    # Word frequency
    word_freq_df = pd.DataFrame(word_freq.most_common(1000), 
                               columns=['Word', 'Frequency'])
    word_freq_df.to_csv(f'{output_dir}/word_frequency.csv', index=False)
    
    # Topic text statistics
    topic_stats.to_csv(f'{output_dir}/topic_text_statistics.csv')
    
    # Detailed segment analysis
    segment_analysis = df[['topic', 'segment_id', 'duration', 'word_count', 
                          'char_count', 'speaking_rate']].copy()
    segment_analysis.to_csv(f'{output_dir}/segment_text_analysis.csv', index=False)
    
    print(f"\nText analysis results saved to {output_dir}/")

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
    print("Analyzing text statistics...")
    df = analyze_text_statistics(segments)
    
    print("Analyzing vocabulary...")
    word_freq, unique_words, total_words = analyze_vocabulary(df)
    
    print("Analyzing text by topic...")
    topic_stats = analyze_text_by_topic(df)
    
    print("\nCreating text visualizations...")
    create_text_visualizations(df, word_freq, output_dir)
    
    print("Saving text analysis reports...")
    save_text_analysis_report(df, word_freq, topic_stats, output_dir)
    
    print("\nText analysis completed!")

if __name__ == "__main__":
    main()
