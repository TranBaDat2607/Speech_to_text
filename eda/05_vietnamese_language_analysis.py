"""
EDA - Vietnamese Language Features Analysis for Speech-to-Text Dataset
Phân tích đặc trưng ngôn ngữ tiếng Việt cho dataset speech-to-text
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

def load_all_segments(data_dir):
    """Load all segments from all topic folders"""
    all_segments = []
    
    data_path = Path(data_dir)
    
    for topic_folder in data_path.iterdir():
        if topic_folder.is_dir():
            topic_name = topic_folder.name
            
            for json_file in topic_folder.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        segment = json.load(f)
                        segment['topic'] = topic_name
                        all_segments.append(segment)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
    
    return all_segments

def analyze_vietnamese_characters(df):
    """Analyze Vietnamese character usage"""
    print("="*60)
    print("VIETNAMESE CHARACTER ANALYSIS")
    print("="*60)
    
    # Combine all text
    all_text = ' '.join(df['text'].fillna(''))
    
    # Vietnamese tone marks and special characters
    vietnamese_chars = {
        'a_variations': ['a', 'à', 'á', 'ả', 'ã', 'ạ', 'ă', 'ằ', 'ắ', 'ẳ', 'ẵ', 'ặ', 'â', 'ầ', 'ấ', 'ẩ', 'ẫ', 'ậ'],
        'e_variations': ['e', 'è', 'é', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ề', 'ế', 'ể', 'ễ', 'ệ'],
        'i_variations': ['i', 'ì', 'í', 'ỉ', 'ĩ', 'ị'],
        'o_variations': ['o', 'ò', 'ó', 'ỏ', 'õ', 'ọ', 'ô', 'ồ', 'ố', 'ổ', 'ỗ', 'ộ', 'ơ', 'ờ', 'ớ', 'ở', 'ỡ', 'ợ'],
        'u_variations': ['u', 'ù', 'ú', 'ủ', 'ũ', 'ụ', 'ư', 'ừ', 'ứ', 'ử', 'ữ', 'ự'],
        'y_variations': ['y', 'ỳ', 'ý', 'ỷ', 'ỹ', 'ỵ'],
        'd_variations': ['d', 'đ']
    }
    
    # Count character frequencies
    char_freq = Counter(all_text.lower())
    
    print(f"VIETNAMESE CHARACTER USAGE:")
    for category, chars in vietnamese_chars.items():
        category_count = sum(char_freq.get(char, 0) for char in chars)
        print(f"{category.replace('_', ' ').title()}: {category_count:,} occurrences")
    
    # Tone mark analysis
    tone_marks = {
        'no_tone': r'[aeiouydăâêôơư]',
        'acute': r'[áéíóúýắấếốớứ]',      # dấu sắc
        'grave': r'[àèìòùỳằầềồờừ]',      # dấu huyền
        'hook': r'[ảẻỉỏủỷẳẩểổởử]',       # dấu hỏi
        'tilde': r'[ãẽĩõũỹẵẫễỗỡữ]',      # dấu ngã
        'dot': r'[ạẹịọụỵặậệộợự]'         # dấu nặng
    }
    
    print(f"\nTONE MARK DISTRIBUTION:")
    total_vowels = 0
    for tone_name, pattern in tone_marks.items():
        count = len(re.findall(pattern, all_text.lower()))
        total_vowels += count
        print(f"{tone_name.replace('_', ' ').title()}: {count:,} ({count/len(all_text)*100:.2f}%)")
    
    return char_freq, vietnamese_chars

def analyze_vietnamese_phonetics(df):
    """Analyze Vietnamese phonetic patterns"""
    print(f"\nVIETNAMESE PHONETIC PATTERNS:")
    
    all_text = ' '.join(df['text'].fillna(''))
    
    # Common Vietnamese syllable patterns
    syllable_patterns = {
        'consonant_clusters': [
            'ch', 'gh', 'gi', 'kh', 'ng', 'nh', 'ph', 'qu', 'th', 'tr'
        ],
        'final_consonants': [
            'c', 'ch', 'm', 'n', 'ng', 'nh', 'p', 't'
        ]
    }
    
    # Count consonant clusters
    print(f"\nCONSONANT CLUSTERS:")
    for cluster in syllable_patterns['consonant_clusters']:
        count = len(re.findall(rf'\b{cluster}', all_text.lower()))
        print(f"{cluster}: {count:,} occurrences")
    
    # Analyze syllable structure
    words = all_text.lower().split()
    
    # Count syllables (rough estimation)
    total_syllables = 0
    syllable_lengths = []
    
    for word in words:
        # Remove punctuation
        clean_word = re.sub(r'[^\w\u00C0-\u1EF9]', '', word)
        if clean_word:
            # Count vowel groups as syllables
            vowel_groups = re.findall(r'[aeiouăâêôơưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]+', clean_word)
            syllable_count = len(vowel_groups)
            total_syllables += syllable_count
            syllable_lengths.append(syllable_count)
    
    print(f"\nSYLLABLE ANALYSIS:")
    print(f"Total syllables: {total_syllables:,}")
    print(f"Average syllables per word: {np.mean(syllable_lengths):.2f}")
    print(f"Monosyllabic words: {syllable_lengths.count(1)} ({syllable_lengths.count(1)/len(syllable_lengths)*100:.1f}%)")
    
    return syllable_patterns, syllable_lengths

def analyze_vietnamese_vocabulary(df):
    """Analyze Vietnamese vocabulary characteristics"""
    print(f"\nVIETNAMESE VOCABULARY ANALYSIS:")
    
    all_text = ' '.join(df['text'].fillna(''))
    words = re.findall(r'\b[\w\u00C0-\u1EF9]+\b', all_text.lower())
    
    word_freq = Counter(words)
    
    # Common Vietnamese function words
    function_words = [
        'là', 'của', 'có', 'được', 'và', 'với', 'trong', 'cho', 'để', 'từ',
        'về', 'theo', 'như', 'khi', 'nếu', 'mà', 'hay', 'hoặc', 'nhưng',
        'tôi', 'bạn', 'chúng', 'họ', 'nó', 'này', 'đó', 'những', 'các'
    ]
    
    # Count function words
    function_word_count = sum(word_freq.get(word, 0) for word in function_words)
    total_words = len(words)
    
    print(f"Total words: {total_words:,}")
    print(f"Unique words: {len(word_freq):,}")
    print(f"Function words: {function_word_count:,} ({function_word_count/total_words*100:.1f}%)")
    
    # Most common words
    print(f"\nTOP 20 MOST COMMON VIETNAMESE WORDS:")
    for word, count in word_freq.most_common(20):
        percentage = (count / total_words) * 100
        print(f"{word}: {count:,} ({percentage:.2f}%)")
    
    # Word length analysis
    word_lengths = [len(word) for word in words]
    
    print(f"\nWORD LENGTH ANALYSIS:")
    print(f"Average word length: {np.mean(word_lengths):.2f} characters")
    print(f"Median word length: {np.median(word_lengths):.1f} characters")
    print(f"Most common word length: {Counter(word_lengths).most_common(1)[0][0]} characters")
    
    return word_freq, word_lengths, function_words

def analyze_transcription_quality(df):
    """Analyze transcription quality specific to Vietnamese"""
    print(f"\nTRANSCRIPTION QUALITY ANALYSIS:")
    
    quality_issues = []
    
    # 1. Mixed case issues (Vietnamese typically uses lowercase)
    mixed_case_segments = 0
    for text in df['text'].fillna(''):
        if re.search(r'[a-zA-Z\u00C0-\u1EF9]', text):  # Has letters
            if text != text.lower() and text != text.upper():
                mixed_case_segments += 1
    
    if mixed_case_segments > 0:
        quality_issues.append(f"Mixed case segments: {mixed_case_segments}")
        print(f"- Mixed case segments: {mixed_case_segments}")
    
    # 2. Missing tone marks (potential transcription errors)
    missing_tones = 0
    for text in df['text'].fillna(''):
        words = text.lower().split()
        for word in words:
            # Check if word has Vietnamese vowels but no tone marks
            if re.search(r'[aeiouăâêôơưy]', word) and not re.search(r'[àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]', word):
                # This might be a word missing tone marks
                if len(word) > 2:  # Ignore very short words
                    missing_tones += 1
                    break
    
    if missing_tones > 0:
        quality_issues.append(f"Segments potentially missing tone marks: {missing_tones}")
        print(f"- Segments potentially missing tone marks: {missing_tones}")
    
    # 3. Non-Vietnamese characters
    non_vietnamese_chars = set()
    for text in df['text'].fillna(''):
        for char in text:
            if char.isalpha() and not re.match(r'[a-zA-Z\u00C0-\u1EF9]', char):
                non_vietnamese_chars.add(char)
    
    if non_vietnamese_chars:
        quality_issues.append(f"Non-Vietnamese characters found: {len(non_vietnamese_chars)}")
        print(f"- Non-Vietnamese characters found: {list(non_vietnamese_chars)[:10]}")
    
    # 4. Repeated words (potential transcription errors)
    repeated_words = 0
    for text in df['text'].fillna(''):
        words = text.lower().split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 2:
                repeated_words += 1
                break
    
    if repeated_words > 0:
        quality_issues.append(f"Segments with repeated words: {repeated_words}")
        print(f"- Segments with repeated words: {repeated_words}")
    
    return quality_issues

def create_vietnamese_visualizations(df, char_freq, word_freq, syllable_lengths, output_dir):
    """Create Vietnamese language analysis visualizations"""
    plt.style.use('default')
    
    # 1. Vietnamese character frequency
    vietnamese_chars = [char for char in char_freq.keys() 
                       if re.match(r'[a-zA-Z\u00C0-\u1EF9]', char)]
    top_chars = dict(Counter({char: char_freq[char] for char in vietnamese_chars}).most_common(30))
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(top_chars)), list(top_chars.values()))
    plt.title('Top 30 Vietnamese Characters Frequency')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    plt.xticks(range(len(top_chars)), list(top_chars.keys()))
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vietnamese_chars_frequency.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Word length distribution
    word_lengths = [len(word) for word in word_freq.keys()]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(word_lengths, bins=range(1, max(word_lengths)+2), alpha=0.7, edgecolor='black')
    plt.title('Vietnamese Word Length Distribution')
    plt.xlabel('Word Length (characters)')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(syllable_lengths, bins=range(1, max(syllable_lengths)+2), alpha=0.7, edgecolor='black')
    plt.title('Syllables per Word Distribution')
    plt.xlabel('Syllables per Word')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/vietnamese_word_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Top Vietnamese words
    top_words = dict(word_freq.most_common(30))
    
    plt.figure(figsize=(15, 8))
    plt.bar(range(len(top_words)), list(top_words.values()))
    plt.title('Top 30 Most Frequent Vietnamese Words')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(range(len(top_words)), list(top_words.keys()), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_vietnamese_words.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Tone mark distribution
    all_text = ' '.join(df['text'].fillna(''))
    tone_counts = {
        'No tone': len(re.findall(r'[aeiouydăâêôơư]', all_text.lower())),
        'Acute (sắc)': len(re.findall(r'[áéíóúýắấếốớứ]', all_text.lower())),
        'Grave (huyền)': len(re.findall(r'[àèìòùỳằầềồờừ]', all_text.lower())),
        'Hook (hỏi)': len(re.findall(r'[ảẻỉỏủỷẳẩểổởử]', all_text.lower())),
        'Tilde (ngã)': len(re.findall(r'[ãẽĩõũỹẵẫễỗỡữ]', all_text.lower())),
        'Dot (nặng)': len(re.findall(r'[ạẹịọụỵặậệộợự]', all_text.lower()))
    }
    
    plt.figure(figsize=(10, 8))
    plt.pie(tone_counts.values(), labels=tone_counts.keys(), autopct='%1.1f%%')
    plt.title('Vietnamese Tone Mark Distribution')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/tone_mark_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_vietnamese_analysis_report(df, char_freq, word_freq, syllable_lengths, 
                                  quality_issues, function_words, output_dir):
    """Save Vietnamese language analysis results"""
    
    # Character frequency analysis
    vietnamese_chars = {char: freq for char, freq in char_freq.items() 
                       if re.match(r'[a-zA-Z\u00C0-\u1EF9]', char)}
    char_freq_df = pd.DataFrame(list(vietnamese_chars.items()), 
                               columns=['Character', 'Frequency'])
    char_freq_df = char_freq_df.sort_values('Frequency', ascending=False)
    char_freq_df.to_csv(f'{output_dir}/vietnamese_character_frequency.csv', index=False)
    
    # Word frequency analysis
    word_freq_df = pd.DataFrame(word_freq.most_common(1000), 
                               columns=['Word', 'Frequency'])
    word_freq_df.to_csv(f'{output_dir}/vietnamese_word_frequency.csv', index=False)
    
    # Function words analysis
    function_word_stats = []
    total_words = sum(word_freq.values())
    
    for fw in function_words:
        count = word_freq.get(fw, 0)
        percentage = (count / total_words) * 100 if total_words > 0 else 0
        function_word_stats.append({
            'Function_Word': fw,
            'Frequency': count,
            'Percentage': percentage
        })
    
    function_words_df = pd.DataFrame(function_word_stats)
    function_words_df = function_words_df.sort_values('Frequency', ascending=False)
    function_words_df.to_csv(f'{output_dir}/vietnamese_function_words.csv', index=False)
    
    # Language statistics summary
    all_text = ' '.join(df['text'].fillna(''))
    
    language_stats = {
        'Metric': ['Total Characters', 'Total Words', 'Unique Words', 
                  'Average Word Length', 'Average Syllables per Word',
                  'Monosyllabic Words Percentage', 'Function Words Percentage'],
        'Value': [len(all_text), len(all_text.split()), len(word_freq),
                 np.mean([len(word) for word in word_freq.keys()]),
                 np.mean(syllable_lengths),
                 (syllable_lengths.count(1) / len(syllable_lengths)) * 100,
                 (sum(word_freq.get(fw, 0) for fw in function_words) / total_words) * 100]
    }
    
    language_stats_df = pd.DataFrame(language_stats)
    language_stats_df.to_csv(f'{output_dir}/vietnamese_language_statistics.csv', index=False)
    
    # Quality issues report
    with open(f'{output_dir}/vietnamese_transcription_quality.txt', 'w', encoding='utf-8') as f:
        f.write("VIETNAMESE TRANSCRIPTION QUALITY REPORT\n")
        f.write("="*50 + "\n\n")
        if quality_issues:
            for issue in quality_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("No major transcription quality issues detected.\n")
    
    print(f"\nVietnamese language analysis results saved to {output_dir}/")

def main():
    # Configuration
    data_dir = "c:/Users/Admin/Desktop/dat301m/crawl_data/output_segments_grouped"
    output_dir = "c:/Users/Admin/Desktop/dat301m/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    segments = load_all_segments(data_dir)
    df = pd.DataFrame(segments)
    
    print("Analyzing Vietnamese characters...")
    char_freq, vietnamese_chars = analyze_vietnamese_characters(df)
    
    print("Analyzing Vietnamese phonetics...")
    syllable_patterns, syllable_lengths = analyze_vietnamese_phonetics(df)
    
    print("Analyzing Vietnamese vocabulary...")
    word_freq, word_lengths, function_words = analyze_vietnamese_vocabulary(df)
    
    print("Analyzing transcription quality...")
    quality_issues = analyze_transcription_quality(df)
    
    print("\nCreating Vietnamese language visualizations...")
    create_vietnamese_visualizations(df, char_freq, word_freq, syllable_lengths, output_dir)
    
    print("Saving Vietnamese analysis reports...")
    save_vietnamese_analysis_report(df, char_freq, word_freq, syllable_lengths, 
                                   quality_issues, function_words, output_dir)
    
    print("\nVietnamese language analysis completed!")

if __name__ == "__main__":
    main()
