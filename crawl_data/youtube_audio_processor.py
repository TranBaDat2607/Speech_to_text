import os
import yt_dlp
import subprocess
from pathlib import Path
import json
import re
from pydub import AudioSegment
import math
import logging
from retry_utils import retry_on_exception, OperationTracker
from file_validators import FileValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeAudioProcessor:
    def __init__(self, audio_folder="audio", output_folder="audio_segments"):
        self.audio_folder = Path(audio_folder)
        self.output_folder = Path(output_folder)
        self.tracker = OperationTracker("Audio Processing")
        self.create_folders()
        
    def create_folders(self):
        """Create necessary folders"""
        try:
            self.audio_folder.mkdir(exist_ok=True, parents=True)
            self.output_folder.mkdir(exist_ok=True, parents=True)
            logger.info(f"Audio folder: {self.audio_folder.absolute()}")
            logger.info(f"Output folder: {self.output_folder.absolute()}")
        except Exception as e:
            logger.error(f"Error creating folders: {e}")
            raise
        
    def extract_video_id(self, url):
        """Trích xuất video ID từ URL YouTube"""
        if "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return None
        
    def clean_filename(self, filename):
        """Làm sạch tên file"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
        
    @retry_on_exception(max_attempts=3, delay=5.0, backoff=2.0)
    def download_audio(self, youtube_url):
        """Download audio from YouTube video with retry logic"""
        video_id = None
        try:
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                error_msg = f"Cannot extract video ID from: {youtube_url}"
                logger.error(error_msg)
                self.tracker.record_failure(youtube_url, error_msg)
                return None

            logger.info(f"Downloading audio for video ID: {video_id}")
            
            # Cấu hình yt-dlp để tải audio chất lượng cao
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': str(self.audio_folder / f'{video_id}.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Get video info
                    info = ydl.extract_info(youtube_url, download=False)
                    if not info:
                        error_msg = f"Cannot get video info: {youtube_url}"
                        logger.error(error_msg)
                        self.tracker.record_failure(video_id, error_msg)
                        return None

                    title = self.clean_filename(info.get('title', video_id))
                    duration = info.get('duration', 0)

                    logger.info(f"Title: {title}")
                    logger.info(f"Duration: {duration} seconds")

                    # Download audio
                    ydl.download([youtube_url])

                    # Find downloaded audio file
                    audio_file = None
                    for file_path in self.audio_folder.glob(f"{video_id}.*"):
                        if file_path.suffix in ['.wav', '.mp3', '.m4a']:
                            audio_file = file_path
                            break

                    if audio_file and audio_file.exists():
                        # Validate audio file
                        is_valid, error = FileValidator.validate_audio_file(audio_file, min_size=10240)
                        if not is_valid:
                            logger.error(f"Audio validation failed: {error}")
                            self.tracker.record_failure(video_id, f"Invalid audio: {error}")
                            return None

                        logger.info(f"Downloaded audio: {audio_file.name}")
                        self.tracker.record_success(video_id)
                        return {
                            'video_id': video_id,
                            'title': title,
                            'duration': duration,
                            'file_path': audio_file,
                            'url': youtube_url
                        }
                    else:
                        error_msg = f"Audio file not found for {video_id}"
                        logger.error(error_msg)
                        self.tracker.record_failure(video_id, error_msg)
                        return None

                except Exception as e:
                    error_msg = f"Error downloading audio: {e}"
                    logger.error(error_msg, exc_info=True)
                    self.tracker.record_failure(video_id, error_msg)
                    return None
                    
        except Exception as e:
            error_msg = f"Error processing {youtube_url}: {e}"
            logger.error(error_msg, exc_info=True)
            if video_id:
                self.tracker.record_failure(video_id, error_msg)
            return None
            
    def split_audio_to_segments(self, audio_info, segment_duration=30):
        """Split audio into segments with error handling"""
        video_id = audio_info.get('video_id', 'unknown')
        try:
            audio_file = audio_info['file_path']

            logger.info(f"Splitting audio {video_id} into {segment_duration}s segments...")

            # Load audio file
            audio = AudioSegment.from_file(audio_file)

            # Calculate number of segments
            total_duration = len(audio) / 1000  # Convert milliseconds to seconds
            num_segments = math.ceil(total_duration / segment_duration)

            logger.info(f"Total duration: {total_duration:.2f}s, creating {num_segments} segments")
            
            segments_created = []
            
            for i in range(num_segments):
                start_time = i * segment_duration * 1000  # milliseconds
                end_time = min((i + 1) * segment_duration * 1000, len(audio))
                
                segment = audio[start_time:end_time]
                
                # Tạo tên file segment
                segment_filename = f"{video_id}_segment_{i+1:03d}.wav"
                segment_path = self.output_folder / segment_filename
                
                # Export segment
                segment.export(segment_path, format="wav")

                # Validate created segment
                is_valid, error = FileValidator.validate_audio_file(segment_path, min_size=1024)
                if not is_valid:
                    logger.warning(f"Invalid segment created: {segment_filename} - {error}")
                    continue

                segment_info = {
                    'video_id': video_id,
                    'segment_id': i + 1,
                    'filename': segment_filename,
                    'start_time': start_time / 1000,
                    'end_time': end_time / 1000,
                    'duration': (end_time - start_time) / 1000,
                    'file_path': str(segment_path)
                }

                segments_created.append(segment_info)
                logger.debug(f"Created segment {i+1}/{num_segments}: {segment_filename} ({segment_info['duration']:.2f}s)")

            logger.info(f"Successfully created {len(segments_created)} segments for {video_id}")
            return segments_created

        except Exception as e:
            logger.error(f"Error splitting audio for {video_id}: {e}", exc_info=True)
            return []
            
    def process_urls_from_file(self, file_path, limit=3):
        """Process URLs from file with error resilience"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]

            # Remove duplicates and limit
            unique_urls = list(set(urls))[:limit]
            logger.info(f"Processing {len(unique_urls)} videos")

            results = {
                'processed_videos': [],
                'all_segments': [],
                'failed_videos': [],
                'summary': {
                    'total_videos': len(unique_urls),
                    'successful_downloads': 0,
                    'failed_downloads': 0,
                    'total_segments': 0
                }
            }

            for i, url in enumerate(unique_urls, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing video {i}/{len(unique_urls)}: {url}")
                logger.info(f"{'='*60}")

                try:
                    # Download audio
                    audio_info = self.download_audio(url)
                    if audio_info:
                        results['summary']['successful_downloads'] += 1
                        results['processed_videos'].append(audio_info)

                        # Split into segments
                        segments = self.split_audio_to_segments(audio_info)
                        if segments:
                            results['all_segments'].extend(segments)
                            results['summary']['total_segments'] += len(segments)
                            logger.info(f"Successfully created {len(segments)} segments")
                        else:
                            logger.warning("No segments created")
                    else:
                        results['summary']['failed_downloads'] += 1
                        results['failed_videos'].append(url)
                        logger.error(f"Failed to download audio for: {url}")

                except Exception as e:
                    # Continue processing even if one video fails
                    results['summary']['failed_downloads'] += 1
                    results['failed_videos'].append(url)
                    logger.error(f"Error processing video {url}: {e}", exc_info=True)
            
            # Save metadata
            metadata_file = self.output_folder / "processing_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)

            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info("PROCESSING SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Videos successfully processed: {results['summary']['successful_downloads']}/{results['summary']['total_videos']}")
            logger.info(f"Videos failed: {results['summary']['failed_downloads']}")
            logger.info(f"Total segments created: {results['summary']['total_segments']}")
            logger.info(f"Metadata saved: {metadata_file}")

            if results['failed_videos']:
                logger.warning(f"Failed videos ({len(results['failed_videos'])}):")
                for url in results['failed_videos']:
                    logger.warning(f"  - {url}")

            # Print operation tracker summary
            self.tracker.print_summary()

            return results

        except Exception as e:
            logger.error(f"Error processing URL file: {e}", exc_info=True)
            return None

def main():
    """Hàm main để chạy pipeline"""
    url_file = "youtube_video_urls.txt"
    
    if not os.path.exists(url_file):
        print(f"Không tìm thấy file: {url_file}")
        return
    
    print("YouTube Audio Processing Pipeline")
    print("Tải audio từ YouTube và chia thành segments 30 giây")
    
    processor = YouTubeAudioProcessor()
    
    # Xử lý 3 video đầu tiên
    results = processor.process_urls_from_file(url_file, limit=3)
    
    if results:
        print("Pipeline hoàn thành")
        print(f"Kiểm tra thư mục {processor.output_folder} để xem file .wav")
    else:
        print("Pipeline thất bại")

if __name__ == "__main__":
    main()
