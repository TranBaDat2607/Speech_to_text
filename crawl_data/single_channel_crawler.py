#!/usr/bin/env python3
"""
Single YouTube Channel Crawler
Crawl tất cả video từ một YouTube channel cụ thể
"""

import yt_dlp
import json
import time
from pathlib import Path
from typing import List, Dict
import re

class SingleChannelCrawler:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': None,  # Lấy tất cả video
        }
    
    def extract_channel_info(self, channel_url: str) -> Dict:
        """Trích xuất thông tin channel"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                return {
                    'channel_id': info.get('id', ''),
                    'channel_name': info.get('title', 'Unknown Channel'),
                    'channel_url': channel_url,
                    'video_count': len(info.get('entries', []))
                }
        except Exception as e:
            print(f"Loi khi lay thong tin channel: {e}")
            return {}
    
    def get_all_videos_from_channel(self, channel_url: str, max_videos: int = None) -> List[Dict]:
        """Lấy tất cả video từ một channel"""
        try:
            print(f"Dang crawl channel: {channel_url}")
            
            # Cấu hình để lấy tất cả video
            opts = self.ydl_opts.copy()
            if max_videos:
                opts['playlistend'] = max_videos
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Lấy thông tin channel và video
                channel_info = ydl.extract_info(channel_url, download=False)
                
                if not channel_info:
                    print("Khong lay duoc thong tin channel")
                    return []
                
                channel_name = channel_info.get('title', 'Unknown Channel')
                print(f"Channel: {channel_name}")
                
                videos = []
                entries = channel_info.get('entries', [])
                
                print(f"Tim thay {len(entries)} video")
                
                for i, entry in enumerate(entries, 1):
                    if entry and 'id' in entry:
                        video_data = {
                            'video_id': entry['id'],
                            'url': f"https://www.youtube.com/watch?v={entry['id']}",
                            'title': entry.get('title', 'Unknown'),
                            'duration': entry.get('duration', 0),
                            'channel': channel_name,
                            'channel_url': channel_url,
                            'index': i
                        }
                        videos.append(video_data)
                        
                        if i % 10 == 0:
                            print(f"Da xu ly {i}/{len(entries)} video...")
                
                print(f"Hoan thanh crawl {len(videos)} video tu channel {channel_name}")
                return videos
                
        except Exception as e:
            print(f"Loi khi crawl channel: {e}")
            return []
    
    def filter_videos_by_duration(self, videos: List[Dict], min_duration: int = 60, max_duration: int = 3600) -> List[Dict]:
        """Lọc video theo thời lượng (giây)"""
        filtered = []
        
        for video in videos:
            duration = video.get('duration', 0)
            if duration and min_duration <= duration <= max_duration:
                filtered.append(video)
        
        print(f"Loc video theo thoi luong ({min_duration}s - {max_duration}s): {len(filtered)}/{len(videos)}")
        return filtered
    
    def save_video_urls(self, videos: List[Dict], output_file: str = "youtube_video_urls.txt"):
        """Lưu video URLs và metadata"""
        output_path = Path(output_file)
        
        # Lưu URLs
        with open(output_path, 'w', encoding='utf-8') as f:
            for video in videos:
                f.write(f"{video['url']}\n")
        
        # Lưu metadata chi tiết
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)
        
        print(f"Da luu {len(videos)} URLs vao {output_path}")
        print(f"Da luu metadata vao {metadata_file}")
        
        return output_path, metadata_file
    
    def crawl_channel(self, channel_url: str, max_videos: int = None, 
                     filter_duration: bool = True, min_duration: int = 60, max_duration: int = 3600):
        """Crawl một channel hoàn chỉnh"""
        print("Single YouTube Channel Crawler")
        print("="*50)
        
        # Lấy tất cả video
        videos = self.get_all_videos_from_channel(channel_url, max_videos)
        
        if not videos:
            print("Khong lay duoc video nao")
            return None
        
        # Lọc theo thời lượng nếu cần
        if filter_duration:
            videos = self.filter_videos_by_duration(videos, min_duration, max_duration)
        
        if not videos:
            print("Khong co video nao sau khi loc")
            return None
        
        # Lưu kết quả
        urls_file, metadata_file = self.save_video_urls(videos)
        
        # Thống kê
        channel_name = videos[0]['channel'] if videos else 'Unknown'
        total_duration = sum(v.get('duration', 0) for v in videos)
        
        print(f"\nTHONG KE CRAWL:")
        print(f"Channel: {channel_name}")
        print(f"Tong so video: {len(videos)}")
        print(f"Tong thoi luong: {total_duration/3600:.1f} gio")
        print(f"Thoi luong trung binh: {total_duration/len(videos)/60:.1f} phut/video")
        print(f"File URLs: {urls_file}")
        print(f"File metadata: {metadata_file}")
        
        return {
            'videos': videos,
            'urls_file': urls_file,
            'metadata_file': metadata_file,
            'channel_name': channel_name,
            'total_videos': len(videos),
            'total_duration': total_duration
        }

def main():
    """Hàm main để test"""
    crawler = SingleChannelCrawler()
    
    # Ví dụ channel URL
    example_channels = [
        "https://www.youtube.com/@sachnoivietnam15",  # Sách nói
        "https://www.youtube.com/@KhoaiLangThang",     # Travel
        "https://www.youtube.com/@Spiderum",           # Education
    ]
    
    print("Vi du cac channel co the crawl:")
    for i, url in enumerate(example_channels, 1):
        print(f"  {i}. {url}")
    
    print(f"\nDe crawl channel, su dung:")
    print(f"crawler = SingleChannelCrawler()")
    print(f"result = crawler.crawl_channel('CHANNEL_URL')")

if __name__ == "__main__":
    main()
