#!/usr/bin/env python3
"""
Configuration Manager
Quản lý cấu hình từ file JSON cho YouTube crawling pipeline
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any

class ConfigManager:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        """Load cấu hình từ file JSON"""
        if not self.config_file.exists():
            print(f"Config file not found: {self.config_file}")
            print("Creating default config file...")
            self.create_default_config()
            
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Loaded config from: {self.config_file}")
            return config
        except Exception as e:
            print(f"Error reading config: {e}")
            print("Using default config...")
            return self.get_default_config()
    
    def get_default_config(self) -> Dict[str, Any]:
        """Trả về cấu hình mặc định"""
        return {
            "channels": [
                "https://www.youtube.com/@sachnoivietnam15",
                "https://www.youtube.com/@KhoaiLangThang",
                "https://www.youtube.com/@Spiderum"
            ],
            "crawl_settings": {
                "max_videos_per_channel": 20,
                "min_duration_seconds": 60,
                "max_duration_seconds": 3600,
                "video_selection_strategy": "longest",
                "batch_multiplier": 3
            },
            "subtitle_settings": {
                "languages": ["vi", "en"],
                "format": "srt"
            },
            "audio_settings": {
                "segment_duration_seconds": 30,
                "audio_format": "wav",
                "sample_rate": 16000
            },
            "pipeline_settings": {
                "cleanup_intermediate_files": True,
                "create_dataset_summary": True,
                "parallel_processing": False
            },
            "output_settings": {
                "base_output_dir": "datasets",
                "create_channel_folders": True,
                "save_metadata": True
            }
        }
    
    def create_default_config(self):
        """Tạo file config mặc định"""
        default_config = self.get_default_config()
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, ensure_ascii=False, indent=2)
            print(f"Created default config file: {self.config_file}")
        except Exception as e:
            print(f"Error creating config: {e}")
    
    def get_channels(self) -> List[str]:
        """Lấy danh sách channel URLs"""
        return self.config.get('channels', [])
    
    def get_crawl_settings(self) -> Dict[str, Any]:
        """Lấy cài đặt crawl"""
        return self.config.get('crawl_settings', {})
    
    def get_max_videos_per_channel(self) -> int:
        """Lấy số video tối đa mỗi channel"""
        return self.get_crawl_settings().get('max_videos_per_channel', 20)
    
    def get_min_duration(self) -> int:
        """Lấy thời lượng video tối thiểu (giây)"""
        return self.get_crawl_settings().get('min_duration_seconds', 60)
    
    def get_max_duration(self) -> int:
        """Lấy thời lượng video tối đa (giây)"""
        return self.get_crawl_settings().get('max_duration_seconds', 3600)
    
    def get_video_selection_strategy(self) -> str:
        """Lấy chiến lược chọn video: longest, newest, most_viewed"""
        return self.get_crawl_settings().get('video_selection_strategy', 'longest')
    
    def get_batch_multiplier(self) -> int:
        """Lấy hệ số nhân batch size"""
        return self.get_crawl_settings().get('batch_multiplier', 3)
    
    def get_subtitle_languages(self) -> List[str]:
        """Lấy danh sách ngôn ngữ subtitle"""
        return self.config.get('subtitle_settings', {}).get('languages', ['vi', 'en'])
    
    def get_segment_duration(self) -> int:
        """Lấy độ dài audio segment (giây)"""
        return self.config.get('audio_settings', {}).get('segment_duration_seconds', 30)
    
    def get_cleanup_mode(self) -> bool:
        """Lấy chế độ cleanup"""
        return self.config.get('pipeline_settings', {}).get('cleanup_intermediate_files', True)
    
    def get_output_dir(self) -> str:
        """Lấy thư mục output"""
        return self.config.get('output_settings', {}).get('base_output_dir', 'datasets')
    
    def should_create_channel_folders(self) -> bool:
        """Kiểm tra có tạo thư mục riêng cho mỗi channel không"""
        return self.config.get('output_settings', {}).get('create_channel_folders', True)
    
    def print_config_summary(self):
        """In tóm tắt cấu hình"""
        print("\n" + "="*60)
        print("CRAWL PIPELINE CONFIGURATION")
        print("="*60)
        
        channels = self.get_channels()
        print(f"Number of channels: {len(channels)}")
        for i, channel in enumerate(channels, 1):
            print(f"  {i}. {channel}")
        
        crawl_settings = self.get_crawl_settings()
        print(f"\nCrawl settings:")
        print(f"- Max videos/channel: {self.get_max_videos_per_channel()}")
        print(f"- Video duration: {self.get_min_duration()}s - {self.get_max_duration()}s")
        print(f"- Video selection strategy: {self.get_video_selection_strategy()}")
        print(f"- Batch multiplier: {self.get_batch_multiplier()}")
        
        print(f"\nSubtitle settings:")
        print(f"- Languages: {', '.join(self.get_subtitle_languages())}")
        
        print(f"\nAudio settings:")
        print(f"- Segment duration: {self.get_segment_duration()}s")
        
        print(f"\nPipeline settings:")
        print(f"- Cleanup files: {'Yes' if self.get_cleanup_mode() else 'No'}")
        print(f"- Output directory: {self.get_output_dir()}")
        print(f"- Create channel folders: {'Yes' if self.should_create_channel_folders() else 'No'}")
        
        total_videos = len(channels) * self.get_max_videos_per_channel()
        print(f"\nTarget total videos: {total_videos}")
        print("="*60)

def main():
    """Test ConfigManager"""
    config = ConfigManager()
    config.print_config_summary()

if __name__ == "__main__":
    main()
