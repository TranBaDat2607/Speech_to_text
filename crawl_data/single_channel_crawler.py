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
            'extract_flat': False,  # Lấy thông tin chi tiết để kiểm tra video
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
            print(f"Error getting channel info: {e}")
            return {}
    
    def check_video_availability(self, video_url: str) -> bool:
        """Kiểm tra video có khả dụng không"""
        try:
            opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': False,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                # Kiểm tra video có thể truy cập và có thời lượng
                if info and info.get('duration', 0) > 0:
                    return True
                return False
        except:
            return False
    
    def get_valid_videos_from_channel(self, channel_url: str, target_count: int = 20, 
                                     min_duration: int = 30, max_duration: int = float('inf'),
                                     selection_strategy: str = 'first', batch_multiplier: int = 3) -> List[Dict]:
        """Lấy video hợp lệ từ channel với số lượng mục tiêu"""
        try:
            print(f"Crawling channel: {channel_url}")
            print(f"Target: {target_count} valid videos")
            print(f"Duration: {min_duration}s - {max_duration}s")
            print(f"Strategy: {selection_strategy}")
            
            # Bắt đầu với số lượng lớn hơn để có đủ video sau khi lọc
            batch_size = max(target_count * batch_multiplier, 50)
            
            opts = self.ydl_opts.copy()
            opts['playlistend'] = batch_size
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Lấy thông tin channel và video
                channel_info = ydl.extract_info(channel_url, download=False)
                
                if not channel_info:
                    print("Could not get channel info")
                    return []
                
                channel_name = channel_info.get('title', 'Unknown Channel')
                print(f"Channel: {channel_name}")
                
                entries = channel_info.get('entries', [])
                print(f"Found {len(entries)} videos in first batch")
                
                valid_videos = []
                processed = 0
                
                for entry in entries:
                    if len(valid_videos) >= target_count:
                        break
                        
                    processed += 1
                    
                    if not entry or 'id' not in entry:
                        continue
                    
                    video_id = entry['id']
                    video_url = f"https://www.youtube.com/watch?v={video_id}"
                    
                    # Kiểm tra video có hợp lệ không
                    print(f"Checking video {processed}: {video_id}")
                    
                    try:
                        # Lấy thông tin chi tiết video
                        video_info = ydl.extract_info(video_url, download=False)
                        
                        if not video_info:
                            print(f"  -> Could not get video info {video_id}")
                            continue
                            
                        duration = video_info.get('duration', 0)
                        title = video_info.get('title', 'Unknown')
                        
                        # Kiểm tra video có thời lượng hợp lệ
                        if duration < min_duration:
                            print(f"  -> Video too short ({duration}s < {min_duration}s): {title[:50]}")
                            continue
                            
                        if duration > max_duration:
                            print(f"  -> Video too long ({duration}s > {max_duration}s): {title[:50]}")
                            continue
                            
                        # Kiểm tra video không bị private/deleted
                        if video_info.get('availability') in ['private', 'premium_only', 'subscriber_only']:
                            print(f"  -> Video not available: {title[:50]}")
                            continue
                        
                        video_data = {
                            'video_id': video_id,
                            'url': video_url,
                            'title': title,
                            'duration': duration,
                            'channel': channel_name,
                            'channel_url': channel_url,
                            'index': len(valid_videos) + 1
                        }
                        
                        valid_videos.append(video_data)
                        print(f"  -> OK: {title[:50]} ({duration}s)")
                        
                        if len(valid_videos) % 5 == 0:
                            print(f"Got {len(valid_videos)}/{target_count} valid videos")
                            
                    except Exception as e:
                        print(f"  -> Error checking video {video_id}: {e}")
                        continue
                
                # Nếu chưa đủ video, thử lấy thêm
                if len(valid_videos) < target_count and len(entries) >= batch_size:
                    print(f"Only got {len(valid_videos)}/{target_count} videos, trying to get more...")
                    
                    # Tăng batch size và thử lại
                    opts['playlistend'] = batch_size * 2
                    
                    try:
                        extended_info = ydl.extract_info(channel_url, download=False)
                        extended_entries = extended_info.get('entries', [])[batch_size:]
                        
                        for entry in extended_entries:
                            if len(valid_videos) >= target_count:
                                break
                                
                            if not entry or 'id' not in entry:
                                continue
                                
                            video_id = entry['id']
                            video_url = f"https://www.youtube.com/watch?v={video_id}"
                            
                            try:
                                video_info = ydl.extract_info(video_url, download=False)
                                
                                if not video_info:
                                    continue
                                    
                                duration = video_info.get('duration', 0)
                                title = video_info.get('title', 'Unknown')
                                
                                if duration < min_duration or duration > max_duration:
                                    continue
                                    
                                if video_info.get('availability') in ['private', 'premium_only', 'subscriber_only']:
                                    continue
                                
                                video_data = {
                                    'video_id': video_id,
                                    'url': video_url,
                                    'title': title,
                                    'duration': duration,
                                    'channel': channel_name,
                                    'channel_url': channel_url,
                                    'index': len(valid_videos) + 1
                                }
                                
                                valid_videos.append(video_data)
                                print(f"Added video: {title[:50]} ({duration}s)")
                                
                            except:
                                continue
                                
                    except Exception as e:
                        print(f"Error getting more videos: {e}")
                
                # Áp dụng chiến lược sắp xếp video
                if selection_strategy == 'longest' and len(valid_videos) > target_count:
                    print(f"Sorting {len(valid_videos)} videos by duration (descending)...")
                    valid_videos.sort(key=lambda x: x['duration'], reverse=True)
                    valid_videos = valid_videos[:target_count]
                    print(f"Selected {len(valid_videos)} longest videos")
                elif selection_strategy == 'shortest' and len(valid_videos) > target_count:
                    print(f"Sorting {len(valid_videos)} videos by duration (ascending)...")
                    valid_videos.sort(key=lambda x: x['duration'])
                    valid_videos = valid_videos[:target_count]
                    print(f"Selected {len(valid_videos)} shortest videos")
                
                print(f"Completed crawling {len(valid_videos)} valid videos from channel {channel_name}")
                return valid_videos
                
        except Exception as e:
            print(f"Error crawling channel: {e}")
            return []
    
    def filter_videos_by_duration(self, videos: List[Dict], min_duration: int = 60, max_duration: int = 3600) -> List[Dict]:
        """Lọc video theo thời lượng (giây)"""
        filtered = []
        
        for video in videos:
            duration = video.get('duration', 0)
            if duration and min_duration <= duration <= max_duration:
                filtered.append(video)
        
        print(f"Filtered videos by duration ({min_duration}s - {max_duration}s): {len(filtered)}/{len(videos)}")
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
        
        print(f"Saved {len(videos)} URLs to {output_path}")
        print(f"Saved metadata to {metadata_file}")
        
        return output_path, metadata_file
    
    def crawl_channel(self, channel_url: str, max_videos: int = None, 
                     filter_duration: bool = True, min_duration: int = 60, max_duration: int = 3600,
                     selection_strategy: str = 'first', batch_multiplier: int = 3):
        """Crawl một channel hoàn chỉnh với kiểm tra video hợp lệ"""
        print("Single YouTube Channel Crawler")
        print("="*50)
        
        # Sử dụng hàm mới để lấy video hợp lệ
        target_count = max_videos if max_videos else 20
        videos = self.get_valid_videos_from_channel(
            channel_url, target_count, min_duration, max_duration, 
            selection_strategy, batch_multiplier
        )
        
        if not videos:
            print("Could not get any videos")
            return None
        
        # Lọc theo thời lượng nếu cần (áp dụng thêm filter nếu yêu cầu)
        if filter_duration and (min_duration > 30 or max_duration < float('inf')):
            print(f"Applying duration filter: {min_duration}s - {max_duration}s")
            videos = self.filter_videos_by_duration(videos, min_duration, max_duration)
        
        if not videos:
            print("No videos after duration filtering")
            return None
        
        # Lưu kết quả
        urls_file, metadata_file = self.save_video_urls(videos)
        
        # Thống kê
        channel_name = videos[0]['channel'] if videos else 'Unknown'
        total_duration = sum(v.get('duration', 0) for v in videos)
        
        print(f"\nCRAWL STATISTICS:")
        print(f"Channel: {channel_name}")
        print(f"Total videos: {len(videos)}")
        print(f"Total duration: {total_duration/3600:.1f} hours")
        print(f"Average duration: {total_duration/len(videos)/60:.1f} minutes/video")
        print(f"URLs file: {urls_file}")
        print(f"Metadata file: {metadata_file}")
        
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
    print("SingleChannelCrawler - Use in pipeline")
    print("To crawl channel, use:")
    print("crawler = SingleChannelCrawler()")
    print("result = crawler.crawl_channel('CHANNEL_URL')")

if __name__ == "__main__":
    main()
