import os
import yt_dlp
import re
from pathlib import Path

class YouTubeSubtitleDownloader:
    def __init__(self, download_folder="subtitles"):
        self.download_folder = Path(download_folder)
        self.create_download_folder()
        
    def create_download_folder(self):
        """Tạo thư mục để lưu subtitle"""
        self.download_folder.mkdir(exist_ok=True)
        print(f"Thư mục tải xuống: {self.download_folder.absolute()}")
        
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
        
    def download_subtitle(self, youtube_url):
        """Tải subtitle cho một video YouTube bằng yt-dlp"""
        try:
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                print(f"Không thể trích xuất video ID từ: {youtube_url}")
                return False
                
            print(f"Đang xử lý video ID: {video_id}")
            
            # Cấu hình yt-dlp để chỉ tải subtitle
            ydl_opts = {
                'writesubtitles': True,           # Tải subtitle
                'writeautomaticsub': True,       # Tải auto-generated subtitle
                'subtitleslangs': ['vi', 'en'],  # Ưu tiên tiếng Việt và tiếng Anh
                'subtitlesformat': 'srt/best',   # Format SRT
                'skip_download': True,           # Không tải video
                'outtmpl': str(self.download_folder / f'{video_id}.%(ext)s'),
                'ignoreerrors': True,            # Bỏ qua lỗi và tiếp tục
            }
            
            downloaded_files = []
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    # Lấy thông tin video
                    info = ydl.extract_info(youtube_url, download=False)
                    title = self.clean_filename(info.get('title', video_id))
                    
                    print(f"Tiêu đề: {title}")
                    
                    # Kiểm tra subtitle có sẵn
                    subtitles = info.get('subtitles', {})
                    automatic_captions = info.get('automatic_captions', {})
                    
                    if subtitles or automatic_captions:
                        print("Đang tải subtitle...")
                        ydl.download([youtube_url])
                        
                        # Kiểm tra file đã tải
                        for file_path in self.download_folder.glob(f"{video_id}.*"):
                            if file_path.suffix in ['.srt', '.vtt']:
                                downloaded_files.append(file_path.name)
                                print(f"Đã tải: {file_path.name}")
                                
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
                                        print(f"Đã tạo: {txt_path.name}")
                                    except Exception as e:
                                        print(f"Lỗi khi tạo file txt: {e}")
                    else:
                        print("Không tìm thấy subtitle cho video này")
                        return False
                        
                except Exception as e:
                    print(f"Lỗi khi tải subtitle: {e}")
                    return False
                    
            if downloaded_files:
                print(f"Hoàn thành tải subtitle cho video {video_id}: {downloaded_files}")
                return True
            else:
                print(f"Không tải được subtitle nào cho video {video_id}")
                return False
                
        except Exception as e:
            print(f"Lỗi khi xử lý {youtube_url}: {e}")
            return False
            
    def process_url_file(self, file_path):
        """Xử lý file chứa danh sách URL"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
                
            # Loại bỏ URL trùng lặp
            unique_urls = list(set(urls))
            print(f"Tìm thấy {len(urls)} URL, {len(unique_urls)} URL duy nhất")
            
            success_count = 0
            failed_urls = []
            
            for i, url in enumerate(unique_urls, 1):
                print(f"\n--- Xử lý {i}/{len(unique_urls)}: {url} ---")
                if self.download_subtitle(url):
                    success_count += 1
                else:
                    failed_urls.append(url)
                    
            print(f"\nKết quả:")
            print(f"- Thành công: {success_count}/{len(unique_urls)} video")
            print(f"- Thất bại: {len(failed_urls)} video")
            
            if failed_urls:
                print(f"\nCác URL thất bại:")
                for url in failed_urls:
                    print(f"  - {url}")
            
        except Exception as e:
            print(f"Lỗi khi đọc file URL: {e}")

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
