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

class ChannelToDatasetPipeline:
    def __init__(self, channel_url: str, max_videos: int = 20, cleanup: bool = True):
        self.channel_url = channel_url
        self.max_videos = max_videos
        self.cleanup = cleanup
        self.results = {}
        
    def step1_crawl_channel(self):
        """Bước 1: Crawl channel để lấy video URLs"""
        print("\n" + "="*60)
        print("BUOC 1: CRAWL CHANNEL VIDEOS")
        print("="*60)
        
        crawler = SingleChannelCrawler()
        
        # Crawl channel với giới hạn video và lọc thời lượng
        result = crawler.crawl_channel(
            channel_url=self.channel_url,
            max_videos=self.max_videos,
            filter_duration=True,
            min_duration=120,    # Tối thiểu 2 phút
            max_duration=1800    # Tối đa 30 phút
        )
        
        if not result:
            print("Khong crawl duoc video nao")
            return False
            
        self.results['crawl'] = result
        print(f"Da crawl {result['total_videos']} video tu channel")
        return True
        
    def step2_download_subtitles(self):
        """Bước 2: Tải subtitles"""
        print("\n" + "="*60)
        print("BUOC 2: TAI SUBTITLES")
        print("="*60)
        
        if not os.path.exists("youtube_video_urls.txt"):
            print("Khong tim thay file youtube_video_urls.txt")
            return False
            
        downloader = YouTubeSubtitleDownloader()
        downloader.process_url_file("youtube_video_urls.txt")
        
        # Kiểm tra kết quả
        subtitles_folder = Path("subtitles")
        if subtitles_folder.exists():
            srt_files = list(subtitles_folder.glob("*.srt"))
            print(f"Da tai {len(srt_files)} file subtitle")
            self.results['subtitles'] = len(srt_files)
            return len(srt_files) > 0
        return False
        
    def step3_process_audio(self):
        """Bước 3: Xử lý audio"""
        print("\n" + "="*60)
        print("BUOC 3: XU LY AUDIO")
        print("="*60)
        
        processor = YouTubeAudioProcessor()
        results = processor.process_urls_from_file("youtube_video_urls.txt", limit=self.max_videos)
        
        # Kiểm tra kết quả
        segments_folder = Path("audio_segments")
        if segments_folder.exists():
            wav_files = list(segments_folder.glob("*.wav"))
            print(f"Da tao {len(wav_files)} file audio segments")
            self.results['audio_segments'] = len(wav_files)
            return len(wav_files) > 0
        return False
        
    def step4_create_labels(self):
        """Bước 4: Tạo labels"""
        print("\n" + "="*60)
        print("BUOC 4: TAO DATASET LABELS")
        print("="*60)
        
        labeler = DatasetLabeler()
        results = labeler.process_all_videos()
        
        # Kiểm tra kết quả
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            wav_files = list(dataset_folder.glob("*.wav"))
            json_files = list(dataset_folder.glob("*.json"))
            print(f"Da tao {len(wav_files)} file .wav va {len(json_files)} file .json")
            self.results['dataset'] = {
                'wav_files': len(wav_files),
                'json_files': len(json_files)
            }
            return len(wav_files) > 0 and len(json_files) > 0
        return False
        
    def step5_cleanup_intermediate_files(self):
        """Bước 5: Xóa các file trung gian để tiết kiệm dung lượng"""
        if not self.cleanup:
            print("Bo qua cleanup - giu lai tat ca file")
            return True
            
        print("\n" + "="*60)
        print("BUOC 5: CLEANUP - XOA FILE TRUNG GIAN")
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
                    print(f"Da xoa thu muc: {folder_name}/")
                except Exception as e:
                    print(f"Loi khi xoa thu muc {folder_name}: {e}")
        
        # Xóa file
        removed_files = []
        for file_name in files_to_remove:
            file_path = Path(file_name)
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files.append(file_name)
                    print(f"Da xoa file: {file_name}")
                except Exception as e:
                    print(f"Loi khi xoa file {file_name}: {e}")
        
        # Tính dung lượng dataset còn lại
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            for file_path in dataset_folder.rglob("*"):
                if file_path.is_file():
                    total_size_after += file_path.stat().st_size
        
        # Thống kê
        size_saved = total_size_before - total_size_after
        
        print(f"\nTHONG KE CLEANUP:")
        print(f"Thu muc da xoa: {len(removed_folders)} ({', '.join(removed_folders)})")
        print(f"File da xoa: {len(removed_files)} ({', '.join(removed_files)})")
        print(f"Dung luong tiet kiem: {size_saved / (1024*1024):.1f} MB")
        print(f"Dung luong dataset con lai: {total_size_after / (1024*1024):.1f} MB")
        print(f"Chi giu lai thu muc 'dataset/' voi file .wav va .json")
        
        return True
        
    def show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        print("\n" + "="*60)
        print("KET QUA CUOI CUNG")
        print("="*60)
        
        # Thông tin channel
        if 'crawl' in self.results:
            crawl_info = self.results['crawl']
            print(f"Channel: {crawl_info['channel_name']}")
            print(f"Video crawled: {crawl_info['total_videos']}")
            print(f"Tong thoi luong: {crawl_info['total_duration']/3600:.1f} gio")
        
        # Thống kê pipeline
        print(f"\nThong ke pipeline:")
        print(f"- Subtitles tai: {self.results.get('subtitles', 0)} file")
        print(f"- Audio segments: {self.results.get('audio_segments', 0)} file")
        
        if 'dataset' in self.results:
            dataset_info = self.results['dataset']
            print(f"- Dataset WAV: {dataset_info['wav_files']} file")
            print(f"- Dataset JSON: {dataset_info['json_files']} file")
        
        # Hiển thị mẫu dataset
        dataset_folder = Path("dataset")
        if dataset_folder.exists():
            wav_files = list(dataset_folder.glob("*.wav"))
            json_files = list(dataset_folder.glob("*.json"))
            
            print(f"\nDataset folder: {dataset_folder.absolute()}")
            
            if wav_files:
                print(f"\nVi du file .wav:")
                for i, wav_file in enumerate(wav_files[:3]):
                    print(f"  {i+1}. {wav_file.name}")
                if len(wav_files) > 3:
                    print(f"  ... va {len(wav_files) - 3} file khac")
            
            # Hiển thị mẫu JSON
            if json_files:
                print(f"\nVi du noi dung JSON:")
                import json
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        sample_json = json.load(f)
                    print(f"File: {json_files[0].name}")
                    print(json.dumps(sample_json, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"Khong doc duoc file JSON: {e}")
                    
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
                print("LOI: Khong crawl duoc channel")
                return False
                
            # Bước 2: Tải subtitles
            if not self.step2_download_subtitles():
                print("LOI: Khong tai duoc subtitle")
                return False
                
            # Bước 3: Xử lý audio  
            if not self.step3_process_audio():
                print("LOI: Khong xu ly duoc audio")
                return False
                
            # Bước 4: Tạo labels
            if not self.step4_create_labels():
                print("LOI: Khong tao duoc labels")
                return False
                
            # Bước 5: Cleanup (nếu được bật)
            if not self.step5_cleanup_intermediate_files():
                print("LOI: Khong cleanup duoc")
                return False
                
            # Hiển thị kết quả
            self.show_final_results()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"\nPIPELINE HOAN THANH!")
            print(f"Thoi gian xu ly: {duration/60:.1f} phut")
            print(f"Dataset san sang cho training model!")
            
            return True
            
        except Exception as e:
            print(f"LOI TRONG PIPELINE: {e}")
            return False

def main():
    """Hàm main với input channel URL"""
    print("YouTube Channel to Dataset Pipeline")
    print("="*50)
    
    # Ví dụ channel URLs
    example_channels = {
        "1": "https://www.youtube.com/@sachnoivietnam15",  # Sách nói Việt Nam
        "2": "https://www.youtube.com/@KhoaiLangThang",     # Travel vlog
        "3": "https://www.youtube.com/@Spiderum",           # Education
        "4": "https://www.youtube.com/@betterversionvn",    # Self-improvement
    }
    
    print("Vi du cac channel co the xu ly:")
    for key, url in example_channels.items():
        print(f"  {key}. {url}")
    
    # Input từ user
    print(f"\nNhap channel URL hoac chon so (1-{len(example_channels)}):")
    user_input = input().strip()
    
    # Xử lý input
    if user_input in example_channels:
        channel_url = example_channels[user_input]
        print(f"Da chon: {channel_url}")
    elif user_input.startswith("https://"):
        channel_url = user_input
    else:
        print("Input khong hop le")
        return
    
    # Nhập số video tối đa
    print("Nhap so video toi da (mac dinh 20):")
    max_videos_input = input().strip()
    
    try:
        max_videos = int(max_videos_input) if max_videos_input else 20
    except ValueError:
        max_videos = 20
    
    print(f"Se xu ly toi da {max_videos} video tu channel")
    
    # Hỏi về cleanup
    print("Co muon xoa file trung gian sau khi hoan thanh? (y/n, mac dinh y):")
    cleanup_input = input().strip().lower()
    cleanup = cleanup_input != 'n'
    
    if cleanup:
        print("Se xoa cac thu muc audio/, audio_segments/, subtitles/ sau khi hoan thanh")
    else:
        print("Se giu lai tat ca file trung gian")
    
    # Chạy pipeline
    pipeline = ChannelToDatasetPipeline(channel_url, max_videos, cleanup)
    success = pipeline.run_pipeline()
    
    if success:
        print("\nHuong dan su dung dataset:")
        print("1. File .wav: Audio 30 giay cho training")
        print("2. File .json: Labels tuong ung voi transcript")
        print("3. File dataset_summary.json: Thong tin tong quan")
    else:
        print("\nPipeline that bai. Kiem tra loi o tren.")

if __name__ == "__main__":
    main()
