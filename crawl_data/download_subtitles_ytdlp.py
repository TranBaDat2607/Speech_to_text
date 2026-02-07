import os
import yt_dlp
import re
from pathlib import Path
import logging
from retry_utils import retry_on_exception, OperationTracker
from file_validators import FileValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class YouTubeSubtitleDownloader:
    def __init__(self, download_folder=None, languages=None, subtitle_format=None):
        if download_folder is None:
            raise ValueError("Download folder must be provided. Cannot use default folder.")
        if languages is None:
            raise ValueError("Subtitle languages must be provided. Cannot use default languages.")
        if subtitle_format is None:
            raise ValueError("Subtitle format must be provided. Cannot use default format.")
        self.download_folder = Path(download_folder)
        self.languages = languages
        self.subtitle_format = subtitle_format
        self.tracker = OperationTracker("Subtitle Download")
        self.create_download_folder()
        
    def create_download_folder(self):
        """Create folder for subtitle downloads"""
        try:
            self.download_folder.mkdir(exist_ok=True, parents=True)
            logger.info(f"Download folder: {self.download_folder.absolute()}")
        except Exception as e:
            logger.error(f"Error creating download folder: {e}")
            raise
        
    def extract_video_id(self, url):
        """Trích xuất video ID từ URL YouTube"""
        if "watch?v=" in url:
            return url.split("watch?v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        return None
        
    def clean_filename(self, filename):
        """Làm sạch tên file để tránh ký tự không hợp lệ"""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
        
    @retry_on_exception(max_attempts=3, delay=3.0, backoff=2.0)
    def download_subtitle(self, youtube_url):
        """Download subtitle for a YouTube video with retry logic"""
        video_id = None
        try:
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                error_msg = f"Cannot extract video ID from: {youtube_url}"
                logger.error(error_msg)
                self.tracker.record_failure(youtube_url, error_msg)
                return False

            logger.info(f"Processing video ID: {video_id}")

            # Cấu hình yt-dlp để chỉ tải subtitle
            ydl_opts = {
                'writesubtitles': True,           # Tải subtitle
                'writeautomaticsub': True,       # Tải auto-generated subtitle
                'subtitleslangs': ['all'],        # Get all available languages
                'subtitlesformat': f'{self.subtitle_format}/best',   # Format from config
                'skip_download': True,           # Không tải video
                'outtmpl': str(self.download_folder / f'{video_id}.%(ext)s'),
                'ignoreerrors': True,            # Bỏ qua lỗi và tiếp tục
                'extractor_args': {'youtube': {'player_client': ['android', 'web']}},  # Use multiple clients
            }

            downloaded_files = []

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Get video info
                    info = ydl.extract_info(youtube_url, download=False)
                    title = self.clean_filename(info.get('title', video_id))

                    logger.info(f"Title: {title}")

                    # Check subtitle availability
                    subtitles = info.get('subtitles', {})
                    automatic_captions = info.get('automatic_captions', {})

                    # Find available languages that match our preferences
                    available_langs = []
                    all_subs = {**automatic_captions, **subtitles}  # Merge both

                    if all_subs:
                        logger.info(f"Available subtitle languages: {list(all_subs.keys())}")

                        # Try to find matching languages in priority order
                        for lang in self.languages:
                            if lang in all_subs:
                                available_langs.append(lang)
                                logger.info(f"Found subtitle for language: {lang}")

                        # If no preferred languages, use the first available
                        if not available_langs and all_subs:
                            first_lang = list(all_subs.keys())[0]
                            available_langs.append(first_lang)
                            logger.info(f"No preferred languages found, using: {first_lang}")

                    if available_langs:
                        # Update ydl_opts with specific languages
                        ydl_opts['subtitleslangs'] = available_langs

                        # Create new YoutubeDL instance with updated options
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl2:
                            logger.info(f"Downloading subtitles for languages: {available_langs}")
                            ydl2.download([youtube_url])

                        # Check downloaded files and validate
                        for file_path in self.download_folder.glob(f"{video_id}.*"):
                            if file_path.suffix in ['.srt', '.vtt']:
                                # Validate SRT file
                                is_valid, error = FileValidator.validate_srt_file(file_path)
                                if not is_valid:
                                    logger.warning(f"Invalid subtitle file: {file_path.name} - {error}")
                                    continue

                                downloaded_files.append(file_path.name)
                                logger.info(f"Downloaded: {file_path.name}")

                                # Tạo bản sao .txt từ .srt
                                if file_path.suffix == '.srt':
                                    txt_path = file_path.with_suffix('.txt')
                                    try:
                                        with open(file_path, 'r', encoding='utf-8') as srt_file:
                                            content = srt_file.read()
                                            # Loại bỏ timestamp và giữ lại text
                                            lines = content.split('\n')
                                            text_lines = []
                                            for line in lines:
                                                line = line.strip()
                                                if line and not line.isdigit() and '-->' not in line:
                                                    text_lines.append(line)

                                        with open(txt_path, 'w', encoding='utf-8') as txt_file:
                                            txt_file.write('\n'.join(text_lines))

                                        downloaded_files.append(txt_path.name)
                                        logger.info(f"Created: {txt_path.name}")
                                    except Exception as e:
                                        logger.warning(f"Error creating txt file: {e}")
                    else:
                        error_msg = "No subtitles found for this video"
                        logger.warning(error_msg)
                        self.tracker.record_failure(video_id, error_msg)
                        return False

                except Exception as e:
                    error_msg = f"Error downloading subtitle: {e}"
                    logger.error(error_msg, exc_info=True)
                    self.tracker.record_failure(video_id, error_msg)
                    return False

            if downloaded_files:
                logger.info(f"Completed subtitle download for video {video_id}: {downloaded_files}")
                self.tracker.record_success(video_id)
                return True
            else:
                error_msg = f"No subtitle files downloaded for video {video_id}"
                logger.warning(error_msg)
                self.tracker.record_failure(video_id, error_msg)
                return False

        except Exception as e:
            error_msg = f"Error processing {youtube_url}: {e}"
            logger.error(error_msg, exc_info=True)
            if video_id:
                self.tracker.record_failure(video_id, error_msg)
            return False
            
    def process_url_file(self, file_path):
        """Process file containing URL list with error resilience"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]

            # Remove duplicates
            unique_urls = list(set(urls))
            logger.info(f"Found {len(urls)} URLs, {len(unique_urls)} unique URLs")

            success_count = 0
            failed_urls = []

            for i, url in enumerate(unique_urls, 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing {i}/{len(unique_urls)}: {url}")
                logger.info(f"{'='*60}")

                try:
                    if self.download_subtitle(url):
                        success_count += 1
                    else:
                        failed_urls.append(url)
                except Exception as e:
                    # Continue processing even if one subtitle fails
                    logger.error(f"Error processing {url}: {e}", exc_info=True)
                    failed_urls.append(url)

            # Print summary
            logger.info(f"\n{'='*60}")
            logger.info("SUBTITLE DOWNLOAD SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Successful: {success_count}/{len(unique_urls)} videos")
            logger.info(f"Failed: {len(failed_urls)} videos")

            if failed_urls:
                logger.warning(f"\nFailed URLs ({len(failed_urls)}):")
                for url in failed_urls[:10]:  # Show first 10
                    logger.warning(f"  - {url}")
                if len(failed_urls) > 10:
                    logger.warning(f"  ... and {len(failed_urls) - 10} more")

            # Print operation tracker summary
            self.tracker.print_summary()

        except Exception as e:
            logger.error(f"Error reading URL file: {e}", exc_info=True)

def main():
    # Đường dẫn đến file chứa URL
    url_file = "youtube_video_urls.txt"
    
    if not os.path.exists(url_file):
        print(f"Không tìm thấy file: {url_file}")
        return
        
    print("=== YouTube Subtitle Downloader với yt-dlp ===")
    print("Công cụ này sẽ tải subtitle trực tiếp từ YouTube")
    print("Không cần trình duyệt web, nhanh và ổn định hơn\n")
    
    downloader = YouTubeSubtitleDownloader()
    downloader.process_url_file(url_file)
    
    print("\nHoàn thành!")

if __name__ == "__main__":
    main()
