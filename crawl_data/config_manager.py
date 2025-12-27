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
            error_msg = f"Config file not found: {self.config_file}. Please create config.json from config.json.example"
            print(f"ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            print(f"Loaded config from: {self.config_file}")
            return config
        except json.JSONDecodeError as e:
            error_msg = f"Invalid JSON in config file {self.config_file}: {e}"
            print(f"ERROR: {error_msg}")
            raise
        except Exception as e:
            error_msg = f"Error reading config file {self.config_file}: {e}"
            print(f"ERROR: {error_msg}")
            raise
    
    def get_channels(self) -> List[str]:
        """Lấy danh sách channel URLs"""
        if 'channels' not in self.config:
            raise KeyError("Missing 'channels' in config.json")
        return self.config['channels']
    
    def get_crawl_settings(self) -> Dict[str, Any]:
        """Lấy cài đặt crawl"""
        if 'crawl_settings' not in self.config:
            raise KeyError("Missing 'crawl_settings' in config.json")
        return self.config['crawl_settings']
    
    def get_max_videos_per_channel(self) -> int:
        """Lấy số video tối đa mỗi channel"""
        settings = self.get_crawl_settings()
        if 'max_videos_per_channel' not in settings:
            raise KeyError("Missing 'max_videos_per_channel' in crawl_settings")
        return settings['max_videos_per_channel']

    def get_min_duration(self) -> int:
        """Lấy thời lượng video tối thiểu (giây)"""
        settings = self.get_crawl_settings()
        if 'min_duration_seconds' not in settings:
            raise KeyError("Missing 'min_duration_seconds' in crawl_settings")
        return settings['min_duration_seconds']

    def get_max_duration(self) -> int:
        """Lấy thời lượng video tối đa (giây)"""
        settings = self.get_crawl_settings()
        if 'max_duration_seconds' not in settings:
            raise KeyError("Missing 'max_duration_seconds' in crawl_settings")
        return settings['max_duration_seconds']

    def get_video_selection_strategy(self) -> str:
        """Lấy chiến lược chọn video: longest, newest, most_viewed"""
        settings = self.get_crawl_settings()
        if 'video_selection_strategy' not in settings:
            raise KeyError("Missing 'video_selection_strategy' in crawl_settings")
        return settings['video_selection_strategy']

    def get_batch_multiplier(self) -> int:
        """Lấy hệ số nhân batch size"""
        settings = self.get_crawl_settings()
        if 'batch_multiplier' not in settings:
            raise KeyError("Missing 'batch_multiplier' in crawl_settings")
        return settings['batch_multiplier']
    
    def get_subtitle_languages(self) -> List[str]:
        """Lấy danh sách ngôn ngữ subtitle"""
        if 'subtitle_settings' not in self.config:
            raise KeyError("Missing 'subtitle_settings' in config.json")
        if 'languages' not in self.config['subtitle_settings']:
            raise KeyError("Missing 'languages' in subtitle_settings")
        return self.config['subtitle_settings']['languages']

    def get_segment_duration(self) -> int:
        """Lấy độ dài audio segment (giây)"""
        if 'audio_settings' not in self.config:
            raise KeyError("Missing 'audio_settings' in config.json")
        if 'segment_duration_seconds' not in self.config['audio_settings']:
            raise KeyError("Missing 'segment_duration_seconds' in audio_settings")
        return self.config['audio_settings']['segment_duration_seconds']

    def get_cleanup_mode(self) -> bool:
        """Lấy chế độ cleanup"""
        if 'pipeline_settings' not in self.config:
            raise KeyError("Missing 'pipeline_settings' in config.json")
        if 'cleanup_intermediate_files' not in self.config['pipeline_settings']:
            raise KeyError("Missing 'cleanup_intermediate_files' in pipeline_settings")
        return self.config['pipeline_settings']['cleanup_intermediate_files']

    def get_output_dir(self) -> str:
        """Lấy thư mục output"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'base_output_dir' not in self.config['output_settings']:
            raise KeyError("Missing 'base_output_dir' in output_settings")
        return self.config['output_settings']['base_output_dir']

    def should_create_channel_folders(self) -> bool:
        """Kiểm tra có tạo thư mục riêng cho mỗi channel không"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'create_channel_folders' not in self.config['output_settings']:
            raise KeyError("Missing 'create_channel_folders' in output_settings")
        return self.config['output_settings']['create_channel_folders']

    def get_subtitles_folder(self) -> str:
        """Lấy tên thư mục subtitles"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'subtitles_folder' not in self.config['output_settings']:
            raise KeyError("Missing 'subtitles_folder' in output_settings")
        return self.config['output_settings']['subtitles_folder']

    def get_audio_folder(self) -> str:
        """Lấy tên thư mục audio"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'audio_folder' not in self.config['output_settings']:
            raise KeyError("Missing 'audio_folder' in output_settings")
        return self.config['output_settings']['audio_folder']

    def get_audio_segments_folder(self) -> str:
        """Lấy tên thư mục audio segments"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'audio_segments_folder' not in self.config['output_settings']:
            raise KeyError("Missing 'audio_segments_folder' in output_settings")
        return self.config['output_settings']['audio_segments_folder']

    def get_dataset_folder(self) -> str:
        """Lấy tên thư mục dataset"""
        if 'output_settings' not in self.config:
            raise KeyError("Missing 'output_settings' in config.json")
        if 'dataset_folder' not in self.config['output_settings']:
            raise KeyError("Missing 'dataset_folder' in output_settings")
        return self.config['output_settings']['dataset_folder']

    def get_subtitle_format(self) -> str:
        """Lấy định dạng subtitle"""
        if 'subtitle_settings' not in self.config:
            raise KeyError("Missing 'subtitle_settings' in config.json")
        if 'format' not in self.config['subtitle_settings']:
            raise KeyError("Missing 'format' in subtitle_settings")
        return self.config['subtitle_settings']['format']

    def get_audio_format(self) -> str:
        """Lấy định dạng audio"""
        if 'audio_settings' not in self.config:
            raise KeyError("Missing 'audio_settings' in config.json")
        if 'audio_format' not in self.config['audio_settings']:
            raise KeyError("Missing 'audio_format' in audio_settings")
        return self.config['audio_settings']['audio_format']

    def get_sample_rate(self) -> int:
        """Lấy sample rate của audio"""
        if 'audio_settings' not in self.config:
            raise KeyError("Missing 'audio_settings' in config.json")
        if 'sample_rate' not in self.config['audio_settings']:
            raise KeyError("Missing 'sample_rate' in audio_settings")
        return self.config['audio_settings']['sample_rate']

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
