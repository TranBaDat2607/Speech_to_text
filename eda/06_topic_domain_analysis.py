"""
EDA - Topic and Domain Analysis for Speech-to-Text Dataset
Phân tích chủ đề và domain cho dataset speech-to-text tiếng Việt
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

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

def analyze_topic_statistics(df):
    """Analyze basic statistics by topic"""
    print("="*60)
    print("TOPIC AND DOMAIN ANALYSIS")
    print("="*60)
    
    topic_stats = df.groupby('topic').agg({
        'segment_id': 'count',
        'duration': ['sum', 'mean', 'std'],
        'text': lambda x: ' '.join(x.fillna('')).split().__len__()
    }).round(2)
    
    topic_stats.columns = ['Segments', 'Total_Duration', 'Mean_Duration', 'Std_Duration', 'Total_Words']
    topic_stats['Duration_Hours'] = (topic_stats['Total_Duration'] / 3600).round(2)
    topic_stats['Words_Per_Segment'] = (topic_stats['Total_Words'] / topic_stats['Segments']).round(2)
    
    # Sort by total duration
    topic_stats = topic_stats.sort_values('Total_Duration', ascending=False)
    
    print(f"TOPIC STATISTICS OVERVIEW:")
    print(f"Total topics: {len(topic_stats)}")
    print(f"Total segments: {topic_stats['Segments'].sum()}")
    print(f"Total duration: {topic_stats['Total_Duration'].sum()/3600:.2f} hours")
    print(f"Total words: {topic_stats['Total_Words'].sum():,}")
    
    print(f"\nTOP 10 TOPICS BY DURATION:")
    print(topic_stats.head(10)[['Segments', 'Duration_Hours', 'Words_Per_Segment']].to_string())
    
    return topic_stats

def analyze_topic_vocabulary(df):
    """Analyze vocabulary characteristics by topic"""
    print(f"\nTOPIC VOCABULARY ANALYSIS:")
    
    topic_vocab = {}
    
    for topic in df['topic'].unique():
        topic_text = ' '.join(df[df['topic'] == topic]['text'].fillna(''))
        
        # Clean text
        topic_text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', topic_text.lower())
        words = topic_text.split()
        
        topic_vocab[topic] = {
            'total_words': len(words),
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / len(words) if len(words) > 0 else 0,
            'most_common': Counter(words).most_common(10)
        }
    
    # Create vocabulary comparison
    vocab_comparison = []
    for topic, stats in topic_vocab.items():
        vocab_comparison.append({
            'Topic': topic,
            'Total_Words': stats['total_words'],
            'Unique_Words': stats['unique_words'],
            'Vocabulary_Richness': stats['vocabulary_richness']
        })
    
    vocab_df = pd.DataFrame(vocab_comparison)
    vocab_df = vocab_df.sort_values('Vocabulary_Richness', ascending=False)
    
    print(f"VOCABULARY RICHNESS BY TOPIC (top 10):")
    print(vocab_df.head(10).to_string(index=False))
    
    return topic_vocab, vocab_df

def calculate_topic_similarity(df):
    """Calculate similarity between topics using TF-IDF"""
    print(f"\nTOPIC SIMILARITY ANALYSIS:")
    
    # Prepare topic documents
    topic_documents = []
    topic_names = []
    
    for topic in df['topic'].unique():
        topic_text = ' '.join(df[df['topic'] == topic]['text'].fillna(''))
        # Clean text
        topic_text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', topic_text.lower())
        topic_documents.append(topic_text)
        topic_names.append(topic)
    
    # Calculate TF-IDF
    try:
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(topic_documents)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find most similar topic pairs
        similarity_pairs = []
        for i in range(len(topic_names)):
            for j in range(i+1, len(topic_names)):
                similarity_pairs.append({
                    'Topic_1': topic_names[i],
                    'Topic_2': topic_names[j],
                    'Similarity': similarity_matrix[i][j]
                })
        
        similarity_df = pd.DataFrame(similarity_pairs)
        similarity_df = similarity_df.sort_values('Similarity', ascending=False)
        
        print(f"MOST SIMILAR TOPIC PAIRS (top 10):")
        print(similarity_df.head(10).to_string(index=False))
        
        print(f"\nLEAST SIMILAR TOPIC PAIRS (bottom 5):")
        print(similarity_df.tail(5).to_string(index=False))
        
        return similarity_matrix, similarity_df, topic_names
        
    except Exception as e:
        print(f"Error calculating topic similarity: {e}")
        return None, None, topic_names

def analyze_domain_characteristics(df, topic_vocab):
    """Analyze domain-specific characteristics"""
    print(f"\nDOMAIN CHARACTERISTICS ANALYSIS:")
    
    # Categorize topics by potential domains
    domain_keywords = {
        'self_development': ['phát triển', 'bản thân', 'tự', 'cải thiện', 'kỹ năng', 'học'],
        'psychology': ['tâm lý', 'cảm xúc', 'tinh thần', 'tư duy', 'suy nghĩ'],
        'health_wellness': ['sức khỏe', 'cơ thể', 'chữa', 'lành', 'khỏe'],
        'relationships': ['tình yêu', 'quan hệ', 'người', 'bạn', 'gia đình'],
        'philosophy': ['triết', 'sống', 'đời', 'cuộc', 'ý nghĩa', 'giá trị'],
        'business_finance': ['tiền', 'kinh doanh', 'công việc', 'thành công']
    }
    
    topic_domains = {}
    
    for topic in df['topic'].unique():
        topic_text = ' '.join(df[df['topic'] == topic]['text'].fillna('').str.lower())
        
        domain_scores = {}
        for domain, keywords in domain_keywords.items():
            score = sum(topic_text.count(keyword) for keyword in keywords)
            domain_scores[domain] = score
        
        # Assign primary domain
        primary_domain = max(domain_scores, key=domain_scores.get) if max(domain_scores.values()) > 0 else 'general'
        topic_domains[topic] = {
            'primary_domain': primary_domain,
            'domain_scores': domain_scores
        }
    
    # Count topics by domain
    domain_counts = Counter(info['primary_domain'] for info in topic_domains.values())
    
    print(f"TOPICS BY DOMAIN:")
    for domain, count in domain_counts.most_common():
        print(f"{domain.replace('_', ' ').title()}: {count} topics")
    
    return topic_domains, domain_counts

def analyze_content_complexity(df):
    """Analyze content complexity by topic"""
    print(f"\nCONTENT COMPLEXITY ANALYSIS:")
    
    complexity_metrics = []
    
    for topic in df['topic'].unique():
        topic_data = df[df['topic'] == topic]
        
        # Calculate various complexity metrics
        texts = topic_data['text'].fillna('')
        
        # Average sentence length
        total_sentences = 0
        total_words = 0
        
        for text in texts:
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            total_sentences += len(sentences)
            total_words += len(text.split())
        
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        
        # Average word length
        all_words = ' '.join(texts).split()
        avg_word_length = np.mean([len(word) for word in all_words]) if all_words else 0
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(all_words))
        vocab_diversity = unique_words / len(all_words) if all_words else 0
        
        # Speaking rate
        total_duration = topic_data['duration'].sum()
        speaking_rate = len(all_words) / total_duration if total_duration > 0 else 0
        
        complexity_metrics.append({
            'Topic': topic,
            'Avg_Sentence_Length': avg_sentence_length,
            'Avg_Word_Length': avg_word_length,
            'Vocab_Diversity': vocab_diversity,
            'Speaking_Rate': speaking_rate,
            'Total_Words': len(all_words),
            'Unique_Words': unique_words
        })
    
    complexity_df = pd.DataFrame(complexity_metrics)
    complexity_df = complexity_df.sort_values('Vocab_Diversity', ascending=False)
    
    print(f"CONTENT COMPLEXITY BY TOPIC (top 10):")
    print(complexity_df.head(10)[['Topic', 'Avg_Sentence_Length', 'Avg_Word_Length', 
                                 'Vocab_Diversity', 'Speaking_Rate']].round(2).to_string(index=False))
    
    return complexity_df

def create_topic_visualizations(topic_stats, vocab_df, similarity_matrix, topic_names, 
                               domain_counts, complexity_df, output_dir):
    """Create topic analysis visualizations"""
    plt.style.use('default')
    
    # 1. Topic distribution by segments and duration
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Segments per topic
    top_topics = topic_stats.head(15)
    axes[0,0].barh(range(len(top_topics)), top_topics['Segments'])
    axes[0,0].set_yticks(range(len(top_topics)))
    axes[0,0].set_yticklabels(top_topics.index, fontsize=8)
    axes[0,0].set_title('Top 15 Topics by Number of Segments')
    axes[0,0].set_xlabel('Number of Segments')
    
    # Duration per topic
    axes[0,1].barh(range(len(top_topics)), top_topics['Duration_Hours'])
    axes[0,1].set_yticks(range(len(top_topics)))
    axes[0,1].set_yticklabels(top_topics.index, fontsize=8)
    axes[0,1].set_title('Top 15 Topics by Duration (Hours)')
    axes[0,1].set_xlabel('Duration (Hours)')
    
    # Words per segment
    axes[1,0].barh(range(len(top_topics)), top_topics['Words_Per_Segment'])
    axes[1,0].set_yticks(range(len(top_topics)))
    axes[1,0].set_yticklabels(top_topics.index, fontsize=8)
    axes[1,0].set_title('Top 15 Topics by Words per Segment')
    axes[1,0].set_xlabel('Words per Segment')
    
    # Vocabulary richness
    top_vocab = vocab_df.head(15)
    axes[1,1].barh(range(len(top_vocab)), top_vocab['Vocabulary_Richness'])
    axes[1,1].set_yticks(range(len(top_vocab)))
    axes[1,1].set_yticklabels(top_vocab['Topic'], fontsize=8)
    axes[1,1].set_title('Top 15 Topics by Vocabulary Richness')
    axes[1,1].set_xlabel('Vocabulary Richness')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/topic_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Topic similarity heatmap
    if similarity_matrix is not None:
        plt.figure(figsize=(15, 12))
        
        # Select top 20 topics for better visualization
        top_20_indices = list(range(min(20, len(topic_names))))
        similarity_subset = similarity_matrix[np.ix_(top_20_indices, top_20_indices)]
        topic_names_subset = [topic_names[i] for i in top_20_indices]
        
        sns.heatmap(similarity_subset, 
                   xticklabels=[name[:20] + '...' if len(name) > 20 else name for name in topic_names_subset],
                   yticklabels=[name[:20] + '...' if len(name) > 20 else name for name in topic_names_subset],
                   annot=False, cmap='viridis', square=True)
        plt.title('Topic Similarity Matrix (Top 20 Topics)')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/topic_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Domain distribution
    plt.figure(figsize=(10, 8))
    domains = list(domain_counts.keys())
    counts = list(domain_counts.values())
    
    plt.pie(counts, labels=[d.replace('_', ' ').title() for d in domains], autopct='%1.1f%%')
    plt.title('Topic Distribution by Domain')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/domain_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Content complexity analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Vocabulary diversity vs Speaking rate
    axes[0,0].scatter(complexity_df['Speaking_Rate'], complexity_df['Vocab_Diversity'], alpha=0.7)
    axes[0,0].set_xlabel('Speaking Rate (words/second)')
    axes[0,0].set_ylabel('Vocabulary Diversity')
    axes[0,0].set_title('Vocabulary Diversity vs Speaking Rate')
    
    # Average word length distribution
    axes[0,1].hist(complexity_df['Avg_Word_Length'], bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].set_xlabel('Average Word Length')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Distribution of Average Word Length by Topic')
    
    # Average sentence length distribution
    axes[1,0].hist(complexity_df['Avg_Sentence_Length'], bins=20, alpha=0.7, edgecolor='black')
    axes[1,0].set_xlabel('Average Sentence Length')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Distribution of Average Sentence Length by Topic')
    
    # Total words vs Unique words
    axes[1,1].scatter(complexity_df['Total_Words'], complexity_df['Unique_Words'], alpha=0.7)
    axes[1,1].set_xlabel('Total Words')
    axes[1,1].set_ylabel('Unique Words')
    axes[1,1].set_title('Total Words vs Unique Words by Topic')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/content_complexity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_topic_analysis_report(topic_stats, vocab_df, similarity_df, topic_domains, 
                              domain_counts, complexity_df, output_dir):
    """Save topic analysis results to files"""
    
    # Topic statistics
    topic_stats.to_csv(f'{output_dir}/topic_statistics.csv')
    
    # Vocabulary analysis
    vocab_df.to_csv(f'{output_dir}/topic_vocabulary_analysis.csv', index=False)
    
    # Topic similarity
    if similarity_df is not None:
        similarity_df.to_csv(f'{output_dir}/topic_similarity_analysis.csv', index=False)
    
    # Domain classification
    domain_classification = []
    for topic, info in topic_domains.items():
        domain_classification.append({
            'Topic': topic,
            'Primary_Domain': info['primary_domain'],
            **{f'Score_{domain}': score for domain, score in info['domain_scores'].items()}
        })
    
    domain_df = pd.DataFrame(domain_classification)
    domain_df.to_csv(f'{output_dir}/topic_domain_classification.csv', index=False)
    
    # Content complexity
    complexity_df.to_csv(f'{output_dir}/topic_content_complexity.csv', index=False)
    
    # Summary report
    with open(f'{output_dir}/topic_analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("TOPIC AND DOMAIN ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Total topics analyzed: {len(topic_stats)}\n")
        f.write(f"Total segments: {topic_stats['Segments'].sum()}\n")
        f.write(f"Total duration: {topic_stats['Total_Duration'].sum()/3600:.2f} hours\n")
        f.write(f"Average segments per topic: {topic_stats['Segments'].mean():.1f}\n")
        f.write(f"Average duration per topic: {topic_stats['Total_Duration'].mean()/60:.1f} minutes\n\n")
        
        f.write("DOMAIN DISTRIBUTION:\n")
        for domain, count in domain_counts.most_common():
            percentage = (count / len(topic_stats)) * 100
            f.write(f"- {domain.replace('_', ' ').title()}: {count} topics ({percentage:.1f}%)\n")
        
        f.write(f"\nMOST COMPLEX TOPICS (by vocabulary diversity):\n")
        for _, row in complexity_df.head(5).iterrows():
            f.write(f"- {row['Topic']}: {row['Vocab_Diversity']:.3f}\n")
        
        f.write(f"\nFASTEST SPEAKING RATE TOPICS:\n")
        fastest_topics = complexity_df.nlargest(5, 'Speaking_Rate')
        for _, row in fastest_topics.iterrows():
            f.write(f"- {row['Topic']}: {row['Speaking_Rate']:.2f} words/second\n")
    
    print(f"\nTopic analysis results saved to {output_dir}/")

def main():
    # Configuration
    data_dir = "c:/Users/Admin/Desktop/dat301m/crawl_data/output_segments_grouped"
    output_dir = "c:/Users/Admin/Desktop/dat301m/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    segments = load_all_segments(data_dir)
    df = pd.DataFrame(segments)
    
    print("Analyzing topic statistics...")
    topic_stats = analyze_topic_statistics(df)
    
    print("Analyzing topic vocabulary...")
    topic_vocab, vocab_df = analyze_topic_vocabulary(df)
    
    print("Calculating topic similarity...")
    similarity_matrix, similarity_df, topic_names = calculate_topic_similarity(df)
    
    print("Analyzing domain characteristics...")
    topic_domains, domain_counts = analyze_domain_characteristics(df, topic_vocab)
    
    print("Analyzing content complexity...")
    complexity_df = analyze_content_complexity(df)
    
    print("\nCreating topic visualizations...")
    create_topic_visualizations(topic_stats, vocab_df, similarity_matrix, topic_names,
                               domain_counts, complexity_df, output_dir)
    
    print("Saving topic analysis reports...")
    save_topic_analysis_report(topic_stats, vocab_df, similarity_df, topic_domains,
                              domain_counts, complexity_df, output_dir)
    
    print("\nTopic and domain analysis completed!")

if __name__ == "__main__":
    main()
