#!/usr/bin/env python3
"""
Channel to Dataset Pipeline
Pipeline hoàn chỉnh từ YouTube channel URL đến dataset speech-to-text
"""

import os
import sys
import time
import shutil
from pathlib import Path
from single_channel_crawler import SingleChannelCrawler
from download_subtitles_ytdlp import YouTubeSubtitleDownloader
from youtube_audio_processor import YouTubeAudioProcessor
from create_dataset_labels import DatasetLabeler
from config_manager import ConfigManager

class ChannelToDatasetPipeline:
    def __init__(self, channel_url: str, config: ConfigManager = None):
        self.channel_url = channel_url
        self.config = config if config else ConfigManager()
        self.max_videos = self.config.get_max_videos_per_channel()
        self.cleanup = self.config.get_cleanup_mode()
        self.results = {}
        
    def step1_crawl_channel(self):
        """Bước 1: Crawl channel để lấy video URLs"""
        print("\n" + "="*60)
        print("STEP 1: CRAWL CHANNEL VIDEOS")
        print("="*60)
        
        crawler = SingleChannelCrawler()
        
        # Crawl channel với cấu hình từ config
        result = crawler.crawl_channel(
            channel_url=self.channel_url,
            max_videos=self.max_videos,
            filter_duration=False,    # Không lọc thêm vì đã lọc trong get_valid_videos_from_channel
            min_duration=self.config.get_min_duration(),
            max_duration=self.config.get_max_duration(),
            selection_strategy=self.config.get_video_selection_strategy(),
            batch_multiplier=self.config.get_batch_multiplier()
        )
        
        if not result:
            print("Could not crawl any videos")
            return False
            
        self.results['crawl'] = result
        print(f"Crawled {result['total_videos']} videos from channel")
        return True
        
    def step2_download_subtitles(self):
        """Bước 2: Tải subtitles"""
        print("\n" + "="*60)
        print("STEP 2: DOWNLOAD SUBTITLES")
        print("="*60)
        
        if not os.path.exists("youtube_video_urls.txt"):
            print("Could not find file youtube_video_urls.txt")
            return False
            
        downloader = YouTubeSubtitleDownloader()
        downloader.process_url_file("youtube_video_urls.txt")
        
        # Kiểm tra kết quả
        subtitles_folder = Path("subtitles")
        if subtitles_folder.exists():
            srt_files = list(subtitles_folder.glob("*.srt"))
            print(f"Downloaded {len(srt_files)} subtitle files")
            self.results['subtitles'] = len(srt_files)
            return len(srt_files) > 0
        return False
        
    def step3_process_audio(self):
        """Bước 3: Xử lý audio"""
        print("\n" + "="*60)
        print("STEP 3: PROCESS AUDIO")
        print("="*60)
        
        processor = YouTubeAudioProcessor()
        results = processor.process_urls_from_file("youtube_video_urls.txt", limit=self.max_videos)
        
        # Kiểm tra kết quả
        segments_folder = Path("audio_segments")
        if segments_folder.exists():
            wav_files = list(segments_folder.glob("*.wav"))
            print(f"Created {len(wav_files)} audio segment files")
            self.results['audio_segments'] = len(wav_files)
            return len(wav_files) > 0
        return False
        
    def step4_create_labels(self):
        """Bước 4: Tạo labels"""
        print("\n" + "="*60)
        print("STEP 4: CREATE DATASET LABELS")
        print("="*60)
        
        labeler = DatasetLabeler()
        results = labeler.process_all_videos()
        
        # Kiểm tra kết quả
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            wav_files = list(dataset_folder.glob("*.wav"))
            json_files = list(dataset_folder.glob("*.json"))
            print(f"Created {len(wav_files)} .wav files and {len(json_files)} .json files")
            self.results['dataset'] = {
                'wav_files': len(wav_files),
                'json_files': len(json_files)
            }
            return len(wav_files) > 0 and len(json_files) > 0
        return False
        
    def step5_cleanup_intermediate_files(self):
        """Bước 5: Xóa các file trung gian để tiết kiệm dung lượng"""
        if not self.cleanup:
            print("Skipping cleanup - keeping all files")
            return True
            
        print("\n" + "="*60)
        print("STEP 5: CLEANUP - DELETE INTERMEDIATE FILES")
        print("="*60)
        
        folders_to_remove = ["audio", "audio_segments", "subtitles"]
        files_to_remove = ["youtube_video_urls.txt", "youtube_video_urls.json", "processing_metadata.json"]
        
        total_size_before = 0
        total_size_after = 0
        
        # Tính dung lượng trước khi xóa
        for folder_name in folders_to_remove:
            folder_path = Path(folder_name)
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file():
                        total_size_before += file_path.stat().st_size
        
        for file_name in files_to_remove:
            file_path = Path(file_name)
            if file_path.exists():
                total_size_before += file_path.stat().st_size
        
        # Xóa thư mục
        removed_folders = []
        for folder_name in folders_to_remove:
            folder_path = Path(folder_name)
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    removed_folders.append(folder_name)
                    print(f"Deleted folder: {folder_name}/")
                except Exception as e:
                    print(f"Error deleting folder {folder_name}: {e}")
        
        # Xóa file
        removed_files = []
        for file_name in files_to_remove:
            file_path = Path(file_name)
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files.append(file_name)
                    print(f"Deleted file: {file_name}")
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")
        
        # Tính dung lượng dataset còn lại
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            for file_path in dataset_folder.rglob("*"):
                if file_path.is_file():
                    total_size_after += file_path.stat().st_size
        
        # Thống kê
        size_saved = total_size_before - total_size_after
        
        print(f"\nCLEANUP STATISTICS:")
        print(f"Folders deleted: {len(removed_folders)} ({', '.join(removed_folders)})")
        print(f"Files deleted: {len(removed_files)} ({', '.join(removed_files)})")
        print(f"Space saved: {size_saved / (1024*1024):.1f} MB")
        print(f"Dataset size remaining: {total_size_after / (1024*1024):.1f} MB")
        print(f"Only keeping 'dataset/' folder with .wav and .json files")
        
        return True
        
    def show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        # Thông tin channel
        if 'crawl' in self.results:
            crawl_info = self.results['crawl']
            print(f"Channel: {crawl_info['channel_name']}")
            print(f"Videos crawled: {crawl_info['total_videos']}")
            print(f"Total duration: {crawl_info['total_duration']/3600:.1f} hours")
        
        # Thống kê pipeline
        print(f"\nPipeline statistics:")
        print(f"- Subtitles downloaded: {self.results.get('subtitles', 0)} files")
        print(f"- Audio segments: {self.results.get('audio_segments', 0)} files")
        
        if 'dataset' in self.results:
            dataset_info = self.results['dataset']
            print(f"- Dataset WAV: {dataset_info['wav_files']} files")
            print(f"- Dataset JSON: {dataset_info['json_files']} files")
        
        # Hiển thị mẫu dataset
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            wav_files = list(dataset_folder.glob("*.wav"))
            json_files = list(dataset_folder.glob("*.json"))
            
            print(f"\nDataset folder: {dataset_folder.absolute()}")
            
            if wav_files:
                print(f"\nSample .wav files:")
                for i, wav_file in enumerate(wav_files[:3]):
                    print(f"  {i+1}. {wav_file.name}")
                if len(wav_files) > 3:
                    print(f"  ... and {len(wav_files) - 3} more files")
            
            # Hiển thị mẫu JSON
            if json_files:
                print(f"\nSample JSON content:")
                import json
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        sample_json = json.load(f)
                    print(f"File: {json_files[0].name}")
                    print(json.dumps(sample_json, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"Could not read JSON file: {e}")
                    
    def run_pipeline(self):
        """Chạy toàn bộ pipeline"""
        print("YOUTUBE CHANNEL TO SPEECH DATASET PIPELINE")
        print(f"Channel URL: {self.channel_url}")
        print(f"Max videos: {self.max_videos}")
        print("="*60)
        
        start_time = time.time()
        
        try:
            # Bước 1: Crawl channel
            if not self.step1_crawl_channel():
                print("ERROR: Could not crawl channel")
                return False
                
            # Bước 2: Tải subtitles
            if not self.step2_download_subtitles():
                print("ERROR: Could not download subtitles")
                return False
                
            # Bước 3: Xử lý audio  
            if not self.step3_process_audio():
                print("ERROR: Could not process audio")
                return False
                
            # Bước 4: Tạo labels
            if not self.step4_create_labels():
                print("ERROR: Could not create labels")
                return False
                
            # Bước 5: Cleanup (nếu được bật)
            if not self.step5_cleanup_intermediate_files():
                print("ERROR: Could not cleanup")
                return False
                
            # Hiển thị kết quả
            self.show_final_results()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\nPIPELINE COMPLETED!")
            print(f"Processing time: {duration/60:.1f} minutes")
            print(f"Dataset ready for model training!")
            
            return True
            
        except Exception as e:
            print(f"PIPELINE ERROR: {e}")
            return False

def main():
    """Hàm main tự động chạy tất cả channel URLs từ config"""
    print("YouTube Channel to Dataset Pipeline - AUTO MODE")
    print("="*50)
    
    # Load cấu hình từ file JSON
    config = ConfigManager()
    config.print_config_summary()
    
    # Lấy danh sách channels từ config
    channels = config.get_channels()
    
    if not channels:
        print("No channels found in config!")
        return
    
    # Thống kê tổng quá trình
    total_success = 0
    total_failed = 0
    failed_channels = []
    
    print(f"\nStarting processing...")
    print("="*60)
    
    for i, channel_url in enumerate(channels, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING CHANNEL {i}/{len(channels)}")
        print(f"URL: {channel_url}")
        print(f"{'='*60}")
        
        try:
            # Tạo thư mục riêng cho mỗi channel nếu được cấu hình
            if config.should_create_channel_folders():
                channel_name = channel_url.split('@')[-1] if '@' in channel_url else f"channel_{i}"
                channel_folder = Path(f"{config.get_output_dir()}/{channel_name}")
                channel_folder.mkdir(parents=True, exist_ok=True)
                
                # Chuyển vào thư mục channel
                original_cwd = os.getcwd()
                os.chdir(channel_folder)
                
                print(f"Processing in directory: {channel_folder.absolute()}")
            else:
                original_cwd = os.getcwd()
            
            # Chạy pipeline cho channel này với config
            pipeline = ChannelToDatasetPipeline(channel_url, config)
            success = pipeline.run_pipeline()
            
            # Quay lại thư mục gốc
            os.chdir(original_cwd)
            
            if success:
                total_success += 1
                print(f"\nSUCCESS: Channel {i} completed")
            else:
                total_failed += 1
                failed_channels.append((i, channel_url))
                print(f"\nFAILED: Channel {i} encountered error")
                
        except Exception as e:
            total_failed += 1
            failed_channels.append((i, channel_url))
            print(f"\nERROR: Channel {i} - {e}")
            
            # Đảm bảo quay lại thư mục gốc nếu có lỗi
            try:
                os.chdir(original_cwd)
            except:
                pass
        
        # Thông báo tiến độ
        print(f"\nProgress: {i}/{len(channels)} channels processed")
        print(f"Success: {total_success}, Failed: {total_failed}")
        
        # Nghỉ giữa các channel để tránh quá tải
        if i < len(channels):
            print("Waiting 10 seconds before processing next channel...")
            time.sleep(10)
    
    # Báo cáo cuối cùng
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"Total channels: {len(channels)}")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    
    if failed_channels:
        print(f"\nFailed channels:")
        for idx, url in failed_channels:
            print(f"  {idx}. {url}")
    
    if total_success > 0:
        print(f"\nSuccessfully created datasets saved in 'datasets/' directory")
        print(f"Each channel has its own folder containing .wav and .json files")
    
    print(f"\nAll completed!")

def main_interactive():
    """Hàm main với input channel URL (phiên bản cũ)"""
    print("YouTube Channel to Dataset Pipeline - Interactive Mode")
    print("="*50)
    
    # Input từ user
    print("Enter channel URL:")
    channel_url = input().strip()
    
    if not channel_url.startswith("https://"):
        print("Invalid URL")
        return
    
    # Nhập số video tối đa
    print("Enter max videos (default 20):")
    max_videos_input = input().strip()
    
    try:
        max_videos = int(max_videos_input) if max_videos_input else 20
    except ValueError:
        max_videos = 20
    
    print(f"Will process up to {max_videos} videos from channel")
    
    # Hỏi về cleanup
    print("Delete intermediate files after completion? (y/n, default y):")
    cleanup_input = input().strip().lower()
    cleanup = cleanup_input != 'n'
    
    if cleanup:
        print("Will delete audio/, audio_segments/, subtitles/ folders after completion")
    else:
        print("Will keep all intermediate files")
    
    # Chạy pipeline
    config = ConfigManager()
    pipeline = ChannelToDatasetPipeline(channel_url, config)
    success = pipeline.run_pipeline()
    
    if success:
        print("\nDataset usage guide:")
        print("1. .wav files: 30-second audio for training")
        print("2. .json files: Labels corresponding to transcript")
        print("3. dataset_summary.json: Overview information")
    else:
        print("\nPipeline failed. Check errors above.")

if __name__ == "__main__":
    main()
