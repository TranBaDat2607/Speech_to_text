"""
YouTube Channel Video URL Crawler
Crawl video URLs from Vietnamese YouTube channels across diverse topics
"""

import yt_dlp
import json
import time
import random
from typing import List, Dict
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeChannelCrawler:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'playlistend': 25,  # Exactly 25 videos per channel
        }
        
        # Vietnamese YouTube channels across diverse topics
        self.vietnamese_channels = {
            # Education & Knowledge
            "Spiderum": "https://www.youtube.com/@Spiderum",
            "Vietcetera": "https://www.youtube.com/@Vietcetera",
            "Khoahoc.tv": "https://www.youtube.com/@KhoaHoc",
            
            # Technology
            "FPT Shop": "https://www.youtube.com/@FPTShop_",
            "Cellphones.com.vn": "https://www.youtube.com/@CellphoneSOfficial",
            
            # Travel & Culture
            "Khoai Lang Thang": "https://www.youtube.com/@KhoaiLangThang",
            "Quang Vinh Passport": "https://www.youtube.com/channel/UCEtFx9C7d3BDqbVRDHfwDEg",
            
            # News & Current Affairs
            "VTV24": "https://www.youtube.com/@vtv24",
            "VnExpress": "https://www.youtube.com/@vnexpress.official",
            "Thanh Nien": "https://www.youtube.com/@thanhnientvnews",
            
            # Audiobooks & Book Reading
            "Better Version VN": "https://www.youtube.com/@betterversionvn",
            "Sach Noi VN": "https://www.youtube.com/@SachnoiVN82",
            "Sach Noi Vietnam": "https://www.youtube.com/@sachnoivietnam15",
        }
    
    def get_channel_videos(self, channel_url: str, channel_name: str) -> List[Dict]:
        """Get video URLs and metadata from a YouTube channel"""
        try:
            logger.info(f"Crawling channel: {channel_name}")
            
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                # Get channel info and videos
                channel_info = ydl.extract_info(channel_url, download=False)
                
                videos = []
                if 'entries' in channel_info:
                    for entry in channel_info['entries']:
                        if entry and 'id' in entry:
                            video_data = {
                                'video_id': entry['id'],
                                'url': f"https://www.youtube.com/watch?v={entry['id']}",
                                'title': entry.get('title', 'Unknown'),
                                'duration': entry.get('duration', 0),
                                'channel': channel_name,
                                'channel_url': channel_url
                            }
                            videos.append(video_data)
                
                logger.info(f"Found {len(videos)} videos from {channel_name}")
                return videos
                
        except Exception as e:
            logger.error(f"Error crawling {channel_name}: {str(e)}")
            return []
    
    def process_videos(self, videos: List[Dict]) -> List[Dict]:
        """Process all videos without filtering by duration"""
        # Return all videos as-is, no duration filtering
        return videos
    
    def crawl_all_channels(self) -> List[Dict]:
        """Crawl exactly 25 videos from each channel (12 channels = 300 videos total)"""
        all_videos = []
        
        logger.info(f"Starting to crawl 25 videos from each of {len(self.vietnamese_channels)} channels")
        logger.info(f"Target: {len(self.vietnamese_channels) * 25} videos total")
        
        for channel_name, channel_url in self.vietnamese_channels.items():
            # Get videos from channel (limited to 25 by ydl_opts)
            videos = self.get_channel_videos(channel_url, channel_name)
            
            # Process all videos without filtering
            processed_videos = self.process_videos(videos)
            
            # Add all videos from this channel
            all_videos.extend(processed_videos)
            
            logger.info(f"Added {len(processed_videos)} videos from {channel_name}. Total: {len(all_videos)} videos")
            
            # Random delay to be respectful
            time.sleep(random.uniform(2, 5))
        
        logger.info(f"Final result: {len(all_videos)} videos from {len(self.vietnamese_channels)} channels")
        logger.info(f"Average: {len(all_videos)/len(self.vietnamese_channels):.1f} videos per channel")
        return all_videos
    
    def save_video_urls(self, videos: List[Dict], output_file: str = "youtube_video_urls.txt"):
        """Save video URLs to text file"""
        output_path = Path(output_file)
        
        # Save URLs only
        with open(output_path, 'w', encoding='utf-8') as f:
            for video in videos:
                f.write(f"{video['url']}\n")
        
        # Save detailed metadata
        metadata_file = output_path.with_suffix('.json')
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(videos)} URLs to {output_path}")
        logger.info(f"Saved metadata to {metadata_file}")
        
        # Print summary
        channels = set(v['channel'] for v in videos)
        
        print(f"\n=== CRAWLING SUMMARY ===")
        print(f"Total videos: {len(videos)}")
        print(f"Channels covered: {len(channels)}")
        print(f"Average videos per channel: {len(videos)/len(channels):.1f}")
        print(f"Expected total duration: ~{len(videos) * 20 / 60:.0f} hours (assuming 20min avg per video)")
        print(f"URLs saved to: {output_path}")
        print(f"Metadata saved to: {metadata_file}")

def main():
    """Main function to run the crawler"""
    crawler = YouTubeChannelCrawler()
    
    # Crawl 25 videos from each channel
    videos = crawler.crawl_all_channels()
    
    # Save to files
    output_file = Path(__file__).parent / "youtube_video_urls.txt"
    crawler.save_video_urls(videos, str(output_file))

if __name__ == "__main__":
    main()
