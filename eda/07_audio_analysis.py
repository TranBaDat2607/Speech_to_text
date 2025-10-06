"""
EDA - Audio Analysis for Whisper Fine-tuning
Phân tích audio cho fine-tune Whisper model
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import librosa
import soundfile as sf
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

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

def load_segments_with_audio_paths(datasets_root):
    """Load segments and find corresponding audio files"""
    all_segments = []
    audio_files = []
    
    dataset_folders = find_dataset_folders(datasets_root)
    
    if not dataset_folders:
        print(f"No dataset folders found in {datasets_root}")
        return all_segments, audio_files
    
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
                    
                    # Find corresponding audio file in the same directory
                    audio_file = None
                    for ext in ['.wav', '.mp3', '.m4a', '.flac']:
                        potential_audio = dataset_path / f"{json_file.stem}{ext}"
                        if potential_audio.exists():
                            audio_file = str(potential_audio)
                            break
                    
                    segment['audio_file'] = audio_file
                    all_segments.append(segment)
                    
                    if audio_file:
                        audio_files.append(audio_file)
                        
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
    
    return all_segments, audio_files

def analyze_audio_files(audio_files, sample_size=100):
    """Analyze audio file characteristics"""
    print("="*60)
    print("AUDIO FILE ANALYSIS")
    print("="*60)
    
    if not audio_files:
        print("No audio files found!")
        return {}
    
    # Sample audio files for analysis (to avoid long processing time)
    sample_files = audio_files[:min(sample_size, len(audio_files))]
    print(f"Analyzing {len(sample_files)} audio files (sample from {len(audio_files)} total)")
    
    audio_stats = {
        'sample_rates': [],
        'durations': [],
        'file_sizes': [],
        'channels': [],
        'formats': [],
        'snr_estimates': [],
        'energy_levels': []
    }
    
    for i, audio_file in enumerate(sample_files):
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=None)
            
            # Basic info
            audio_stats['sample_rates'].append(sr)
            audio_stats['durations'].append(len(y) / sr)
            audio_stats['file_sizes'].append(os.path.getsize(audio_file) / (1024*1024))  # MB
            
            # Get file format
            file_ext = Path(audio_file).suffix.lower()
            audio_stats['formats'].append(file_ext)
            
            # Audio characteristics
            audio_stats['channels'].append(1 if len(y.shape) == 1 else y.shape[0])
            
            # Energy level (RMS)
            rms_energy = np.sqrt(np.mean(y**2))
            audio_stats['energy_levels'].append(rms_energy)
            
            # Simple SNR estimation (signal vs noise floor)
            # Use bottom 10% of energy as noise estimate
            sorted_energy = np.sort(np.abs(y))
            noise_floor = np.mean(sorted_energy[:int(len(sorted_energy) * 0.1)])
            signal_level = np.mean(sorted_energy[int(len(sorted_energy) * 0.9):])
            snr_estimate = 20 * np.log10(signal_level / (noise_floor + 1e-10))
            audio_stats['snr_estimates'].append(snr_estimate)
            
            if (i + 1) % 20 == 0:
                print(f"Processed {i + 1}/{len(sample_files)} files...")
                
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    return audio_stats

def analyze_whisper_compatibility(segments, audio_stats):
    """Analyze Whisper-specific compatibility"""
    print(f"\nWHISPER COMPATIBILITY ANALYSIS:")
    
    compatibility_issues = []
    
    # 1. Sample rate check (Whisper prefers 16kHz)
    if audio_stats.get('sample_rates'):
        sample_rates = Counter(audio_stats['sample_rates'])
        print(f"Sample rate distribution:")
        for sr, count in sample_rates.most_common():
            percentage = (count / len(audio_stats['sample_rates'])) * 100
            print(f"  {sr}Hz: {count} files ({percentage:.1f}%)")
        
        non_16k = sum(1 for sr in audio_stats['sample_rates'] if sr != 16000)
        if non_16k > 0:
            compatibility_issues.append(f"Non-16kHz audio files: {non_16k}")
    
    # 2. Duration analysis for Whisper (30s chunks)
    if audio_stats.get('durations'):
        durations = audio_stats['durations']
        long_files = sum(1 for d in durations if d > 30)
        short_files = sum(1 for d in durations if d < 1)
        
        print(f"\nDuration analysis:")
        print(f"  Mean duration: {np.mean(durations):.2f}s")
        print(f"  Files > 30s: {long_files} ({long_files/len(durations)*100:.1f}%)")
        print(f"  Files < 1s: {short_files} ({short_files/len(durations)*100:.1f}%)")
        
        if long_files > 0:
            compatibility_issues.append(f"Long audio files (>30s): {long_files}")
        if short_files > 0:
            compatibility_issues.append(f"Very short audio files (<1s): {short_files}")
    
    # 3. Channel analysis (Whisper expects mono)
    if audio_stats.get('channels'):
        stereo_files = sum(1 for ch in audio_stats['channels'] if ch > 1)
        if stereo_files > 0:
            compatibility_issues.append(f"Stereo/multi-channel files: {stereo_files}")
            print(f"  Stereo files found: {stereo_files}")
    
    # 4. Audio quality check
    if audio_stats.get('snr_estimates'):
        low_quality = sum(1 for snr in audio_stats['snr_estimates'] if snr < 10)
        if low_quality > 0:
            compatibility_issues.append(f"Low quality audio (SNR < 10dB): {low_quality}")
            print(f"  Low SNR files: {low_quality}")
    
    return compatibility_issues

def analyze_transcript_tokenization(segments):
    """Analyze transcript tokenization for Whisper"""
    print(f"\nTRANSCRIPT TOKENIZATION ANALYSIS:")
    
    try:
        # Try to import tiktoken for accurate tokenization
        import tiktoken
        
        # Use GPT-2 tokenizer as approximation (Whisper uses similar BPE)
        tokenizer = tiktoken.get_encoding("gpt2")
        
        token_lengths = []
        long_segments = []
        
        for segment in segments:
            text = segment.get('text', '')
            if text:
                # Approximate Whisper token count
                tokens = tokenizer.encode(text)
                token_lengths.append(len(tokens))
                
                # Whisper has ~448 token limit per segment
                if len(tokens) > 400:
                    long_segments.append({
                        'segment_id': segment.get('segment_id'),
                        'token_count': len(tokens),
                        'text_preview': text[:100] + '...'
                    })
        
        print(f"Token length statistics (using tiktoken):")
        print(f"  Mean tokens per segment: {np.mean(token_lengths):.1f}")
        print(f"  Max tokens: {max(token_lengths)}")
        print(f"  Segments > 400 tokens: {len(long_segments)}")
        
        if long_segments:
            print(f"\nLong segments (>400 tokens):")
            for seg in long_segments[:5]:  # Show first 5
                print(f"  {seg['segment_id']}: {seg['token_count']} tokens")
        
        return token_lengths, long_segments
        
    except ImportError:
        print("tiktoken not available, using character-based approximation...")
        
        # Fallback: approximate tokenization using character count
        # Rough estimate: 1 token ≈ 4 characters for Vietnamese
        token_lengths = []
        long_segments = []
        
        for segment in segments:
            text = segment.get('text', '')
            if text:
                # Approximate token count (characters / 4)
                approx_tokens = len(text) // 4
                token_lengths.append(approx_tokens)
                
                if approx_tokens > 400:
                    long_segments.append({
                        'segment_id': segment.get('segment_id'),
                        'token_count': approx_tokens,
                        'text_preview': text[:100] + '...'
                    })
        
        print(f"Token length statistics (character-based approximation):")
        if token_lengths:
            print(f"  Mean tokens per segment: {np.mean(token_lengths):.1f}")
            print(f"  Max tokens: {max(token_lengths)}")
            print(f"  Segments > 400 tokens: {len(long_segments)}")
        else:
            print("  No text data found for tokenization analysis")
        
        return token_lengths, long_segments

def create_audio_visualizations(audio_stats, output_dir):
    """Create audio analysis visualizations"""
    from pathlib import Path
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('default')
    
    # 1. Sample rate distribution
    if audio_stats.get('sample_rates'):
        plt.figure(figsize=(10, 6))
        sample_rates = Counter(audio_stats['sample_rates'])
        rates, counts = zip(*sample_rates.most_common())
        
        plt.bar(range(len(rates)), counts)
        plt.title('Audio Sample Rate Distribution')
        plt.xlabel('Sample Rate (Hz)')
        plt.ylabel('Number of Files')
        plt.xticks(range(len(rates)), [f'{r}Hz' for r in rates], rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / 'sample_rate_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 2. Duration vs File Size
    if audio_stats.get('durations') and audio_stats.get('file_sizes'):
        plt.figure(figsize=(10, 6))
        plt.scatter(audio_stats['durations'], audio_stats['file_sizes'], alpha=0.6)
        plt.title('Audio Duration vs File Size')
        plt.xlabel('Duration (seconds)')
        plt.ylabel('File Size (MB)')
        
        # Add correlation
        correlation = np.corrcoef(audio_stats['durations'], audio_stats['file_sizes'])[0,1]
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_path / 'duration_vs_filesize.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. SNR Distribution
    if audio_stats.get('snr_estimates'):
        plt.figure(figsize=(10, 6))
        plt.hist(audio_stats['snr_estimates'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Signal-to-Noise Ratio Distribution')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(audio_stats['snr_estimates']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(audio_stats["snr_estimates"]):.1f}dB')
        plt.axvline(10, color='orange', linestyle='--', label='Quality Threshold (10dB)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'snr_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Energy Level Distribution
    if audio_stats.get('energy_levels'):
        plt.figure(figsize=(10, 6))
        plt.hist(audio_stats['energy_levels'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Audio Energy Level Distribution')
        plt.xlabel('RMS Energy')
        plt.ylabel('Frequency')
        plt.axvline(np.mean(audio_stats['energy_levels']), color='red', linestyle='--',
                   label=f'Mean: {np.mean(audio_stats["energy_levels"]):.4f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / 'energy_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def save_audio_analysis_report(segments, audio_stats, compatibility_issues, token_lengths, output_dir):
    """Save audio analysis results"""
    from pathlib import Path
    output_path = Path(output_dir)
    
    # Audio statistics summary
    if audio_stats:
        audio_summary = {
            'Metric': [
                'Total Audio Files Analyzed',
                'Mean Sample Rate (Hz)',
                'Mean Duration (s)',
                'Mean File Size (MB)',
                'Mean SNR (dB)',
                'Mean Energy Level'
            ],
            'Value': [
                len(audio_stats.get('sample_rates', [])),
                np.mean(audio_stats.get('sample_rates', [0])),
                np.mean(audio_stats.get('durations', [0])),
                np.mean(audio_stats.get('file_sizes', [0])),
                np.mean(audio_stats.get('snr_estimates', [0])),
                np.mean(audio_stats.get('energy_levels', [0]))
            ]
        }
        
        audio_df = pd.DataFrame(audio_summary)
        audio_df.to_csv(output_path / 'audio_analysis_summary.csv', index=False)
    
    # Whisper compatibility report
    with open(output_path / 'whisper_compatibility_report.txt', 'w', encoding='utf-8') as f:
        f.write("WHISPER COMPATIBILITY ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")
        
        if compatibility_issues:
            f.write("COMPATIBILITY ISSUES FOUND:\n")
            for issue in compatibility_issues:
                f.write(f"- {issue}\n")
        else:
            f.write("No major compatibility issues found.\n")
        
        f.write(f"\nTOKENIZATION ANALYSIS:\n")
        if token_lengths:
            f.write(f"- Mean tokens per segment: {np.mean(token_lengths):.1f}\n")
            f.write(f"- Max tokens per segment: {max(token_lengths)}\n")
            f.write(f"- Segments exceeding 400 tokens: {sum(1 for t in token_lengths if t > 400)}\n")
        
        f.write(f"\nRECOMMENDATIONS:\n")
        f.write("- Ensure all audio files are 16kHz mono\n")
        f.write("- Split long audio files (>30s) into chunks\n")
        f.write("- Filter out very short segments (<1s)\n")
        f.write("- Consider noise reduction for low SNR files\n")
    
    print(f"\nAudio analysis results saved to {output_dir}/")

def main():
    # Configuration
    datasets_root = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/crawl_data/datasets"
    output_dir = "c:/Users/Admin/Desktop/dat301m/Speech_to_text/eda/outputs"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading segments and finding audio files...")
    segments, audio_files = load_segments_with_audio_paths(datasets_root)
    
    if not segments:
        print("No segments found! Please check the datasets directory.")
        return
    
    print(f"Total segments loaded: {len(segments)}")
    print(f"Audio files found: {len(audio_files)}")
    
    if audio_files:
        print("Analyzing audio characteristics...")
        audio_stats = analyze_audio_files(audio_files, sample_size=50)  # Analyze 50 files
        
        print("Checking Whisper compatibility...")
        compatibility_issues = analyze_whisper_compatibility(segments, audio_stats)
        
        print("Creating audio visualizations...")
        create_audio_visualizations(audio_stats, output_dir)
    else:
        print("No audio files found for analysis!")
        audio_stats = {}
        compatibility_issues = []
    
    print("Analyzing transcript tokenization...")
    token_lengths, long_segments = analyze_transcript_tokenization(segments)
    
    print("Saving audio analysis reports...")
    save_audio_analysis_report(segments, audio_stats, compatibility_issues, token_lengths, output_dir)
    
    print("\nAudio analysis for Whisper fine-tuning completed!")
    
    # Summary
    if compatibility_issues:
        print(f"\n⚠️ Found {len(compatibility_issues)} compatibility issues:")
        for issue in compatibility_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ No major compatibility issues found!")

if __name__ == "__main__":
    main()
