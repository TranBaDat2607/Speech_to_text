"""
EDA - Whisper Model Readiness Analysis
Phân tích độ sẵn sàng cho fine-tune Whisper model
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
import unicodedata

def find_dataset_folders(datasets_root):
    """Find all dataset folders that contain JSON files"""
    dataset_folders = []
    datasets_path = Path(datasets_root)
    
    for channel_folder in datasets_path.iterdir():
        if channel_folder.is_dir():
            dataset_subfolder = channel_folder / "dataset"
            if dataset_subfolder.exists() and dataset_subfolder.is_dir():
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
    
    dataset_folders = find_dataset_folders(datasets_root)
    
    if not dataset_folders:
        print(f"No dataset folders found in {datasets_root}")
        return all_segments
    
    print(f"Found {len(dataset_folders)} dataset folders")
    
    for dataset_info in dataset_folders:
        channel_name = dataset_info['channel']
        dataset_path = dataset_info['path']
        
        for json_file in dataset_path.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    segment = json.load(f)
                    segment['topic'] = channel_name
                    segment['segment_id'] = json_file.stem
                    if 'duration' not in segment and 'start' in segment and 'end' in segment:
                        segment['duration'] = segment['end'] - segment['start']
                    all_segments.append(segment)
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return all_segments

def analyze_whisper_format_requirements(segments):
    """Analyze Whisper-specific format requirements"""
    print("="*60)
    print("WHISPER FORMAT REQUIREMENTS ANALYSIS")
    print("="*60)
    
    format_issues = []
    
    # 1. Required fields check
    required_fields = ['start', 'end', 'text']
    missing_fields = {}
    
    for field in required_fields:
        missing_count = sum(1 for seg in segments if field not in seg or seg[field] is None)
        if missing_count > 0:
            missing_fields[field] = missing_count
            format_issues.append(f"Missing {field} field: {missing_count} segments")
    
    print(f"Required fields check:")
    for field in required_fields:
        missing = missing_fields.get(field, 0)
        print(f"  {field}: {len(segments) - missing}/{len(segments)} present ({missing} missing)")
    
    # 2. Timestamp format validation
    timestamp_issues = 0
    negative_durations = 0
    zero_durations = 0
    
    for segment in segments:
        if 'start' in segment and 'end' in segment:
            try:
                start = float(segment['start'])
                end = float(segment['end'])
                duration = end - start
                
                if duration < 0:
                    negative_durations += 1
                elif duration == 0:
                    zero_durations += 1
                    
            except (ValueError, TypeError):
                timestamp_issues += 1
    
    print(f"\nTimestamp validation:")
    print(f"  Invalid timestamps: {timestamp_issues}")
    print(f"  Negative durations: {negative_durations}")
    print(f"  Zero durations: {zero_durations}")
    
    if timestamp_issues > 0:
        format_issues.append(f"Invalid timestamp format: {timestamp_issues} segments")
    if negative_durations > 0:
        format_issues.append(f"Negative durations: {negative_durations} segments")
    
    return format_issues

def analyze_text_preprocessing_needs(segments):
    """Analyze text preprocessing requirements for Whisper"""
    print(f"\nTEXT PREPROCESSING ANALYSIS:")
    
    preprocessing_stats = {
        'empty_text': 0,
        'only_punctuation': 0,
        'mixed_languages': 0,
        'special_characters': 0,
        'numbers_only': 0,
        'very_long_text': 0,
        'unicode_issues': 0
    }
    
    special_char_pattern = re.compile(r'[^\w\s\u00C0-\u1EF9.,!?;:\-\'"()]')
    
    for segment in segments:
        text = segment.get('text', '').strip()
        
        if not text:
            preprocessing_stats['empty_text'] += 1
            continue
        
        # Check for only punctuation
        if re.match(r'^[^\w\u00C0-\u1EF9]+$', text):
            preprocessing_stats['only_punctuation'] += 1
        
        # Check for mixed languages (non-Vietnamese characters)
        if re.search(r'[^\w\s\u00C0-\u1EF9.,!?;:\-\'"()\d]', text):
            preprocessing_stats['mixed_languages'] += 1
        
        # Check for excessive special characters
        special_chars = special_char_pattern.findall(text)
        if len(special_chars) > 5:
            preprocessing_stats['special_characters'] += 1
        
        # Check for numbers only
        if re.match(r'^\d+[\s\d]*$', text):
            preprocessing_stats['numbers_only'] += 1
        
        # Check for very long text (>500 characters)
        if len(text) > 500:
            preprocessing_stats['very_long_text'] += 1
        
        # Check for Unicode normalization issues
        try:
            normalized = unicodedata.normalize('NFC', text)
            if normalized != text:
                preprocessing_stats['unicode_issues'] += 1
        except:
            preprocessing_stats['unicode_issues'] += 1
    
    print(f"Text preprocessing requirements:")
    for issue, count in preprocessing_stats.items():
        percentage = (count / len(segments)) * 100
        print(f"  {issue.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    return preprocessing_stats

def analyze_training_data_splits(segments):
    """Analyze data splitting strategy for training"""
    print(f"\nTRAINING DATA SPLIT ANALYSIS:")
    
    # Analyze by topic/channel
    topic_distribution = Counter(seg['topic'] for seg in segments)
    
    print(f"Topic distribution:")
    total_segments = len(segments)
    for topic, count in topic_distribution.most_common():
        percentage = (count / total_segments) * 100
        print(f"  {topic}: {count} segments ({percentage:.1f}%)")
    
    # Duration distribution by topic
    topic_durations = {}
    for segment in segments:
        topic = segment['topic']
        duration = segment.get('duration', 0)
        if topic not in topic_durations:
            topic_durations[topic] = []
        topic_durations[topic].append(duration)
    
    print(f"\nDuration by topic:")
    for topic, durations in topic_durations.items():
        total_duration = sum(durations)
        print(f"  {topic}: {total_duration/3600:.2f} hours ({len(durations)} segments)")
    
    # Suggest split strategy
    print(f"\nRecommended split strategy:")
    print(f"  - Use stratified split by topic to maintain distribution")
    print(f"  - Train: 80% ({int(total_segments * 0.8)} segments)")
    print(f"  - Validation: 15% ({int(total_segments * 0.15)} segments)")
    print(f"  - Test: 5% ({int(total_segments * 0.05)} segments)")
    
    return topic_distribution, topic_durations

def analyze_computational_requirements(segments):
    """Estimate computational requirements for training"""
    print(f"\nCOMPUTATIONAL REQUIREMENTS ESTIMATION:")
    
    total_segments = len(segments)
    total_duration = sum(seg.get('duration', 0) for seg in segments)
    
    # Estimate based on Whisper fine-tuning benchmarks
    # These are rough estimates based on community experience
    
    print(f"Dataset size:")
    print(f"  Total segments: {total_segments:,}")
    print(f"  Total duration: {total_duration/3600:.2f} hours")
    
    # Memory estimation (rough)
    estimated_memory_gb = max(8, total_segments * 0.001)  # Minimum 8GB
    print(f"\nEstimated requirements:")
    print(f"  GPU Memory: {estimated_memory_gb:.1f}+ GB")
    print(f"  Training time (V100): {total_duration/3600 * 0.5:.1f}-{total_duration/3600 * 2:.1f} hours")
    print(f"  Storage needed: {total_duration/3600 * 100:.0f}+ MB (processed data)")
    
    # Batch size recommendations
    if total_segments < 1000:
        batch_size = "4-8"
    elif total_segments < 5000:
        batch_size = "8-16"
    else:
        batch_size = "16-32"
    
    print(f"  Recommended batch size: {batch_size}")
    
    return {
        'total_segments': total_segments,
        'total_duration_hours': total_duration/3600,
        'estimated_memory_gb': estimated_memory_gb,
        'recommended_batch_size': batch_size
    }

def analyze_whisper_tokenizer_compatibility(segments, sample_size=1000):
    """Analyze compatibility with Whisper's tokenizer"""
    print(f"\nWHISPER TOKENIZER COMPATIBILITY:")
    
    # Sample segments for analysis
    sample_segments = segments[:min(sample_size, len(segments))]
    
    # Character frequency analysis
    all_text = ' '.join(seg.get('text', '') for seg in sample_segments)
    char_freq = Counter(all_text)
    
    # Vietnamese-specific characters
    vietnamese_chars = set('àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ')
    vietnamese_chars.update('ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ')
    
    vietnamese_char_count = sum(count for char, count in char_freq.items() if char in vietnamese_chars)
    total_chars = sum(char_freq.values())
    
    print(f"Character analysis (sample of {len(sample_segments)} segments):")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Vietnamese characters: {vietnamese_char_count:,} ({vietnamese_char_count/total_chars*100:.1f}%)")
    print(f"  Unique characters: {len(char_freq)}")
    
    # Find problematic characters
    problematic_chars = []
    for char, count in char_freq.most_common():
        if not (char.isalnum() or char.isspace() or char in '.,!?;:\-\'"()[]{}' or char in vietnamese_chars):
            problematic_chars.append((char, count))
    
    if problematic_chars:
        print(f"\nProblematic characters found:")
        for char, count in problematic_chars[:10]:  # Show top 10
            print(f"  '{char}' (U+{ord(char):04X}): {count} times")
    
    return char_freq, vietnamese_char_count, problematic_chars

def create_whisper_readiness_visualizations(segments, topic_distribution, preprocessing_stats, output_dir):
    """Create Whisper readiness visualizations"""
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Topic distribution for data splits
    plt.figure(figsize=(12, 8))
    topics = list(topic_distribution.keys())
    counts = list(topic_distribution.values())
    
    plt.pie(counts, labels=topics, autopct='%1.1f%%', startangle=90)
    plt.title('Topic Distribution for Training Data Splits')
    plt.tight_layout()
    plt.savefig(output_path / 'topic_distribution_splits.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Text preprocessing requirements
    plt.figure(figsize=(12, 8))
    issues = list(preprocessing_stats.keys())
    counts = list(preprocessing_stats.values())
    
    plt.barh(range(len(issues)), counts)
    plt.yticks(range(len(issues)), [issue.replace('_', ' ').title() for issue in issues])
    plt.xlabel('Number of Segments')
    plt.title('Text Preprocessing Requirements')
    plt.tight_layout()
    plt.savefig(output_path / 'preprocessing_requirements.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Duration distribution for batch planning
    durations = [seg.get('duration', 0) for seg in segments]
    
    plt.figure(figsize=(12, 6))
    plt.hist(durations, bins=50, alpha=0.7, edgecolor='black')
    plt.title('Duration Distribution for Batch Size Planning')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    plt.axvline(30, color='red', linestyle='--', label='Whisper 30s limit')
    plt.axvline(np.mean(durations), color='green', linestyle='--', 
               label=f'Mean: {np.mean(durations):.1f}s')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'duration_batch_planning.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_whisper_readiness_report(segments, format_issues, preprocessing_stats, 
                                topic_distribution, computational_reqs, output_dir):
    """Save Whisper readiness analysis report"""
    from pathlib import Path
    output_path = Path(output_dir)
    
    # Create comprehensive readiness report
    with open(output_path / 'whisper_readiness_report.txt', 'w', encoding='utf-8') as f:
        f.write("WHISPER FINE-TUNING READINESS REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("DATASET OVERVIEW:\n")
        f.write(f"- Total segments: {len(segments):,}\n")
        f.write(f"- Total duration: {sum(seg.get('duration', 0) for seg in segments)/3600:.2f} hours\n")
        f.write(f"- Topics/Channels: {len(topic_distribution)}\n\n")
        
        f.write("FORMAT COMPATIBILITY:\n")
        if format_issues:
            for issue in format_issues:
                f.write(f"- ❌ {issue}\n")
        else:
            f.write("- ✅ All format requirements met\n")
        f.write("\n")
        
        f.write("PREPROCESSING REQUIREMENTS:\n")
        critical_issues = ['empty_text', 'unicode_issues', 'very_long_text']
        for issue, count in preprocessing_stats.items():
            status = "❌" if issue in critical_issues and count > 0 else "✅"
            f.write(f"- {status} {issue.replace('_', ' ').title()}: {count} segments\n")
        f.write("\n")
        
        f.write("COMPUTATIONAL REQUIREMENTS:\n")
        f.write(f"- Estimated GPU Memory: {computational_reqs['estimated_memory_gb']:.1f}+ GB\n")
        f.write(f"- Recommended batch size: {computational_reqs['recommended_batch_size']}\n")
        f.write(f"- Training duration estimate: {computational_reqs['total_duration_hours']*0.5:.1f}-{computational_reqs['total_duration_hours']*2:.1f} hours\n\n")
        
        f.write("RECOMMENDATIONS:\n")
        f.write("1. Data Preparation:\n")
        f.write("   - Normalize Unicode text (NFC)\n")
        f.write("   - Remove empty or very short segments\n")
        f.write("   - Split segments longer than 30 seconds\n")
        f.write("   - Ensure all audio is 16kHz mono\n\n")
        
        f.write("2. Training Strategy:\n")
        f.write("   - Use stratified split by topic (80/15/5)\n")
        f.write("   - Start with small learning rate (1e-5)\n")
        f.write("   - Monitor validation loss carefully\n")
        f.write("   - Use gradient accumulation if memory limited\n\n")
        
        f.write("3. Quality Assurance:\n")
        f.write("   - Validate timestamp alignment\n")
        f.write("   - Check for data leakage between splits\n")
        f.write("   - Test on held-out speakers if available\n")
    
    # Save detailed statistics
    readiness_stats = {
        'Metric': [
            'Total Segments',
            'Total Duration (hours)',
            'Format Issues',
            'Empty Text Segments',
            'Unicode Issues',
            'Very Long Segments',
            'Topics/Channels',
            'Estimated GPU Memory (GB)'
        ],
        'Value': [
            len(segments),
            sum(seg.get('duration', 0) for seg in segments)/3600,
            len(format_issues),
            preprocessing_stats.get('empty_text', 0),
            preprocessing_stats.get('unicode_issues', 0),
            preprocessing_stats.get('very_long_text', 0),
            len(topic_distribution),
            computational_reqs['estimated_memory_gb']
        ]
    }
    
    readiness_df = pd.DataFrame(readiness_stats)
    readiness_df.to_csv(output_path / 'whisper_readiness_stats.csv', index=False)
    
    print(f"\nWhisper readiness analysis saved to {output_dir}/")

def main():
    # Configuration
    datasets_root = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/crawl_data/datasets"
    output_dir = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading segments for Whisper readiness analysis...")
    segments = load_all_segments(datasets_root)
    
    if not segments:
        print("No segments found! Please check the datasets directory.")
        return
    
    print(f"Total segments loaded: {len(segments)}")
    
    # Analyze Whisper format requirements
    format_issues = analyze_whisper_format_requirements(segments)
    
    # Analyze text preprocessing needs
    preprocessing_stats = analyze_text_preprocessing_needs(segments)
    
    # Analyze training data splits
    topic_distribution, topic_durations = analyze_training_data_splits(segments)
    
    # Analyze computational requirements
    computational_reqs = analyze_computational_requirements(segments)
    
    # Analyze tokenizer compatibility
    char_freq, vietnamese_chars, problematic_chars = analyze_whisper_tokenizer_compatibility(segments)
    
    print("\nCreating Whisper readiness visualizations...")
    create_whisper_readiness_visualizations(segments, topic_distribution, preprocessing_stats, output_dir)
    
    print("Saving Whisper readiness reports...")
    save_whisper_readiness_report(segments, format_issues, preprocessing_stats, 
                                topic_distribution, computational_reqs, output_dir)
    
    print("\nWhisper readiness analysis completed!")
    
    # Final assessment
    critical_issues = len(format_issues) + preprocessing_stats.get('empty_text', 0) + preprocessing_stats.get('unicode_issues', 0)
    
    if critical_issues == 0:
        print("\n🎉 Dataset is ready for Whisper fine-tuning!")
    else:
        print(f"\n⚠️ Found {critical_issues} critical issues that need to be addressed before fine-tuning.")
        print("Check the detailed report for specific recommendations.")

if __name__ == "__main__":
    main()
