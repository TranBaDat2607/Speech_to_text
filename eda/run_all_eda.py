"""
Master EDA Script - Run all EDA analyses for Speech-to-Text Dataset
Script tổng hợp để chạy tất cả các phân tích EDA cho dataset speech-to-text tiếng Việt
"""

import os
import sys
from pathlib import Path
import subprocess
import time

def run_eda_script(script_name, description):
    """Run an EDA script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        # Set environment to use UTF-8 encoding
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = '0'
        
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, text=True, encoding='utf-8',
                              env=env, errors='replace')
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully!")
            if result.stdout:
                print("Output:")
                # Handle potential encoding issues in output
                try:
                    print(result.stdout)
                except UnicodeEncodeError:
                    # Fallback: replace problematic characters
                    safe_output = result.stdout.encode('utf-8', 'replace').decode('utf-8')
                    print(safe_output)
        else:
            print(f"✗ {description} failed!")
            if result.stderr:
                print("Error:")
                try:
                    print(result.stderr)
                except UnicodeEncodeError:
                    # Fallback: replace problematic characters
                    safe_error = result.stderr.encode('utf-8', 'replace').decode('utf-8')
                    print(safe_error)
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False
    
    return True

def create_master_report(output_dir):
    """Create a master EDA report combining all analyses"""
    report_path = Path(output_dir) / "00_MASTER_EDA_REPORT.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("SPEECH-TO-TEXT DATASET - MASTER EDA REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write("This report contains comprehensive Exploratory Data Analysis (EDA) results\n")
        f.write("for the Vietnamese Speech-to-Text dataset.\n\n")
        
        f.write("ANALYSES PERFORMED:\n")
        f.write("-" * 20 + "\n")
        f.write("1. Basic Statistics Analysis (01_basic_statistics.py)\n")
        f.write("   - Dataset overview and segment counts\n")
        f.write("   - Duration statistics by topic\n")
        f.write("   - Data distribution visualizations\n\n")
        
        f.write("2. Text Analysis (02_text_analysis.py)\n")
        f.write("   - Word count and character statistics\n")
        f.write("   - Speaking rate analysis\n")
        f.write("   - Vocabulary analysis and word frequency\n")
        f.write("   - Text statistics by topic\n\n")
        
        f.write("3. Audio Timing Analysis (03_audio_timing_analysis.py)\n")
        f.write("   - Duration distribution analysis\n")
        f.write("   - Speaking rate vs duration correlation\n")
        f.write("   - Timing anomalies detection\n")
        f.write("   - Duration categories breakdown\n\n")
        
        f.write("4. Data Quality Analysis (04_data_quality_analysis.py)\n")
        f.write("   - Data completeness and consistency checks\n")
        f.write("   - Text quality validation\n")
        f.write("   - Video ID reference validation\n")
        f.write("   - Missing data pattern analysis\n\n")
        
        f.write("5. Vietnamese Language Analysis (05_vietnamese_language_analysis.py)\n")
        f.write("   - Vietnamese character and tone mark analysis\n")
        f.write("   - Syllable pattern analysis\n")
        f.write("   - Vietnamese-specific vocabulary analysis\n")
        f.write("   - Transcription quality assessment\n\n")
        
        f.write("6. Topic Domain Analysis (06_topic_domain_analysis.py)\n")
        f.write("   - Topic similarity and clustering\n")
        f.write("   - Domain classification analysis\n")
        f.write("   - Content complexity metrics\n")
        f.write("   - Cross-topic vocabulary comparison\n\n")
        
        f.write("7. Audio Analysis for Whisper (07_audio_analysis.py)\n")
        f.write("   - Audio quality metrics (sample rate, SNR, energy)\n")
        f.write("   - Whisper compatibility checks\n")
        f.write("   - Audio format and duration analysis\n")
        f.write("   - Transcript tokenization analysis\n\n")
        
        f.write("8. Whisper Fine-tuning Readiness (08_whisper_readiness.py)\n")
        f.write("   - Format requirements validation\n")
        f.write("   - Text preprocessing requirements\n")
        f.write("   - Training data split recommendations\n")
        f.write("   - Computational requirements estimation\n\n")
        
        f.write("OUTPUT FILES:\n")
        f.write("-" * 20 + "\n")
        f.write("CSV Files:\n")
        f.write("- basic_statistics.csv: Overall dataset statistics\n")
        f.write("- topic_statistics.csv: Statistics by topic/channel\n")
        f.write("- text_statistics.csv: Text analysis summary\n")
        f.write("- word_frequency.csv: Most frequent words\n")
        f.write("- timing_statistics.csv: Timing analysis summary\n")
        f.write("- segment_text_analysis.csv: Detailed segment text analysis\n")
        f.write("- segment_timing_analysis.csv: Detailed segment timing analysis\n\n")
        
        f.write("Visualization Files:\n")
        f.write("- duration_distribution.png: Duration histogram\n")
        f.write("- segments_per_topic.png: Segments count by topic\n")
        f.write("- duration_by_topic.png: Duration distribution by topic\n")
        f.write("- word_count_distribution.png: Word count histogram\n")
        f.write("- speaking_rate_distribution.png: Speaking rate analysis\n")
        f.write("- top_words_frequency.png: Most frequent words chart\n")
        f.write("- wordcloud.png: Word cloud visualization\n")
        f.write("- duration_analysis.png: Comprehensive duration analysis\n")
        f.write("- speaking_rate_vs_duration.png: Correlation analysis\n\n")
        
        f.write("HOW TO INTERPRET RESULTS:\n")
        f.write("-" * 20 + "\n")
        f.write("1. Check basic_statistics.csv for overall dataset health\n")
        f.write("2. Review topic_statistics.csv to understand data distribution\n")
        f.write("3. Examine timing_anomalies.txt for potential data quality issues\n")
        f.write("4. Use visualizations to understand data patterns\n")
        f.write("5. Review word_frequency.csv for vocabulary insights\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("-" * 20 + "\n")
        f.write("1. Address any timing anomalies found\n")
        f.write("2. Consider data balancing if topics are heavily skewed\n")
        f.write("3. Use insights for model training strategy\n")
        f.write("4. Consider text preprocessing based on vocabulary analysis\n")
    
    print(f"\n✓ Master report created: {report_path}")

def setup_encoding():
    """Setup UTF-8 encoding for the entire process"""
    import sys
    import io
    
    # Set environment variables for UTF-8
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONLEGACYWINDOWSSTDIO'] = '0'
    
    # Try to reconfigure stdout/stderr for UTF-8 if possible
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass  # Ignore if reconfigure is not available

def main():
    # Setup UTF-8 encoding first
    setup_encoding()
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("SPEECH-TO-TEXT DATASET - COMPREHENSIVE EDA")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # List of EDA scripts to run
    eda_scripts = [
        ("01_basic_statistics.py", "Basic Statistics Analysis"),
        ("02_text_analysis.py", "Text Analysis"),
        ("03_audio_timing_analysis.py", "Audio Timing Analysis"),
        ("04_data_quality_analysis.py", "Data Quality Analysis"),
        ("05_vietnamese_language_analysis.py", "Vietnamese Language Analysis"),
        ("06_topic_domain_analysis.py", "Topic Domain Analysis"),
        ("07_audio_analysis.py", "Audio Analysis for Whisper"),
        ("08_whisper_readiness.py", "Whisper Fine-tuning Readiness")
    ]
    
    # Track success/failure
    results = {}
    
    # Run each EDA script
    for script_name, description in eda_scripts:
        script_path = script_dir / script_name
        
        if script_path.exists():
            success = run_eda_script(str(script_path), description)
            results[script_name] = success
        else:
            print(f"✗ Script not found: {script_path}")
            results[script_name] = False
    
    # Create master report
    create_master_report(output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print("EDA EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(results.values())
    total = len(results)
    
    for script_name, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{script_name}: {status}")
    
    print(f"\nOverall: {successful}/{total} analyses completed successfully")
    
    if successful == total:
        print("\n🎉 All EDA analyses completed successfully!")
        print(f"📊 Check the results in: {output_dir}")
        print("📋 Start with: 00_MASTER_EDA_REPORT.txt")
    else:
        print(f"\n⚠️  {total - successful} analyses failed. Check error messages above.")
    
    return successful == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
