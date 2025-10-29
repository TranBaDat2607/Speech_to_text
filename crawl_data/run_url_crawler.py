"""
Simple script to run YouTube URL crawler
Usage example for youtube_channel_crawler.py
"""

from youtube_channel_crawler import YouTubeChannelCrawler
from pathlib import Path
import logging

def main():
    """Run the YouTube URL crawler"""
    print("=== YouTube Video URL Crawler ===")
    print("Crawling Vietnamese YouTube channels for 100 hours of content...")
    print("This may take 10-20 minutes depending on network speed.\n")
    
    # Initialize crawler
    crawler = YouTubeChannelCrawler()
    
    # Crawl videos (25 from each channel)
    videos = crawler.crawl_all_channels()
    
    if videos:
        # Save results
        output_file = Path(__file__).parent / "youtube_video_urls.txt"
        crawler.save_video_urls(videos, str(output_file))
        
        print(f"\nSuccessfully crawled {len(videos)} video URLs")
        print(f"URLs saved to: {output_file}")
        print(f"Metadata saved to: {output_file.with_suffix('.json')}")
        
        # Show sample URLs
        print(f"\nSample URLs:")
        for i, video in enumerate(videos[:5]):
            print(f"  {i+1}. {video['title'][:50]}...")
            print(f"     {video['url']}")
        
        if len(videos) > 5:
            print(f"     ... and {len(videos)-5} more videos")
            
    else:
        print("No videos were crawled. Check your internet connection.")

if __name__ == "__main__":
    main()
