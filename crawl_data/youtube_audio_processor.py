import os
import yt_dlp
import subprocess
from pathlib import Path
import json
import re
from pydub import AudioSegment
import math

class YouTubeAudioProcessor:
    def __init__(self, audio_folder="audio", output_folder="audio_segments"):
        self.audio_folder = Path(audio_folder)
        self.output_folder = Path(output_folder)
        self.create_folders()
        
    def create_folders(self):
        """Tạo thư mục cần thiết"""
        self.audio_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
        print(f"Thư mục audio: {self.audio_folder.absolute()}")
        print(f"Thư mục output: {self.output_folder.absolute()}")
        
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
        
    def download_audio(self, youtube_url):
        """Tải audio từ YouTube video"""
        try:
            video_id = self.extract_video_id(youtube_url)
            if not video_id:
                print(f"Không thể trích xuất video ID từ: {youtube_url}")
                return None
                
            print(f"Đang tải audio cho video ID: {video_id}")
            
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
                    # Lấy thông tin video
                    info = ydl.extract_info(youtube_url, download=False)
                    if not info:
                        print(f"Không thể lấy thông tin video: {youtube_url}")
                        return None
                    
                    title = self.clean_filename(info.get('title', video_id))
                    duration = info.get('duration', 0)
                    
                    print(f"Tiêu đề: {title}")
                    print(f"Thời lượng: {duration} giây")
                    
                    # Tải audio
                    ydl.download([youtube_url])
                    
                    # Tìm file audio đã tải
                    audio_file = None
                    for file_path in self.audio_folder.glob(f"{video_id}.*"):
                        if file_path.suffix in ['.wav', '.mp3', '.m4a']:
                            audio_file = file_path
                            break
                    
                    if audio_file and audio_file.exists():
                        print(f"Đã tải audio: {audio_file.name}")
                        return {
                            'video_id': video_id,
                            'title': title,
                            'duration': duration,
                            'file_path': audio_file,
                            'url': youtube_url
                        }
                    else:
                        print(f"Không tìm thấy file audio cho {video_id}")
                        return None
                        
                except Exception as e:
                    print(f"Lỗi khi tải audio: {e}")
                    return None
                    
        except Exception as e:
            print(f"Lỗi khi xử lý {youtube_url}: {e}")
            return None
            
    def split_audio_to_segments(self, audio_info, segment_duration=30):
        """Chia audio thành các segment 30 giây"""
        try:
            audio_file = audio_info['file_path']
            video_id = audio_info['video_id']
            
            print(f"Đang chia audio {video_id} thành segments {segment_duration}s...")
            
            # Load audio file
            audio = AudioSegment.from_file(audio_file)
            
            # Tính số segment
            total_duration = len(audio) / 1000  # chuyển từ milliseconds sang seconds
            num_segments = math.ceil(total_duration / segment_duration)
            
            print(f"Thời lượng total: {total_duration:.2f}s, sẽ tạo {num_segments} segments")
            
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
                print(f"Tạo segment {i+1}/{num_segments}: {segment_filename} ({segment_info['duration']:.2f}s)")
            
            return segments_created
            
        except Exception as e:
            print(f"Lỗi khi chia audio: {e}")
            return []
            
    def process_urls_from_file(self, file_path, limit=3):
        """Xử lý URLs từ file, giới hạn số lượng"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            # Loại bỏ URL trùng lặp và lấy số lượng giới hạn
            unique_urls = list(set(urls))[:limit]
            print(f"Sẽ xử lý {len(unique_urls)} video đầu tiên")
            
            results = {
                'processed_videos': [],
                'all_segments': [],
                'summary': {
                    'total_videos': len(unique_urls),
                    'successful_downloads': 0,
                    'total_segments': 0
                }
            }
            
            for i, url in enumerate(unique_urls, 1):
                print(f"\n=== Xử lý video {i}/{len(unique_urls)}: {url} ===")
                
                # Tải audio
                audio_info = self.download_audio(url)
                if audio_info:
                    results['summary']['successful_downloads'] += 1
                    results['processed_videos'].append(audio_info)
                    
                    # Chia thành segments
                    segments = self.split_audio_to_segments(audio_info)
                    if segments:
                        results['all_segments'].extend(segments)
                        results['summary']['total_segments'] += len(segments)
                        print(f"Tạo thành công {len(segments)} segments")
                    else:
                        print("Không tạo được segments")
                else:
                    print("Không tải được audio")
            
            # Lưu metadata
            metadata_file = self.output_folder / "processing_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"\nKET QUA TONG KET")
            print(f"Videos xu ly thanh cong: {results['summary']['successful_downloads']}/{results['summary']['total_videos']}")
            print(f"Tong so segments tao ra: {results['summary']['total_segments']}")
            print(f"Metadata da luu: {metadata_file}")
            
            return results
            
        except Exception as e:
            print(f"Lỗi khi xử lý file URLs: {e}")
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
