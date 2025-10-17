# YouTube Audio Dataset Crawler

Pipeline tự động để crawl audio và transcript từ YouTube channels, tạo dataset cho ML speech-to-text training.

## Tính năng chính

- Multi-channel crawling: Crawl nhiều YouTube channels cùng lúc
- Auto subtitle download: Tự động tải subtitle tiếng Việt/tiếng Anh 
- Audio segmentation: Chia audio thành segments 30 giây chuẩn
- Dataset generation: Tạo dataset structured cho ML training
- Smart cleanup: Dọn dẹp file trung gian tự động
- JSON configuration: Cấu hình linh hoạt qua file JSON

## Yêu cầu hệ thống

### Python Dependencies
```bash
pip install yt-dlp pydub requests beautifulsoup4 pathlib
```

### External Dependencies  
- FFmpeg: Bắt buộc cho audio processing
  - Windows: Tải từ https://ffmpeg.org/download.html
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`

## Hướng dẫn sử dụng

### Method 1: Configuration File (Khuyến nghị)

#### Bước 1: Tạo config file
```bash
cp config.json.example config.json
```

#### Bước 2: Chỉnh sửa config.json
```json
{
  "channels": [
    "https://www.youtube.com/@sachnoivietnam15",
    "https://www.youtube.com/@KhoaiLangThang"
  ],
  "crawl_settings": {
    "max_videos_per_channel": 5,
    "min_duration_seconds": 1800,
    "max_duration_seconds": 7200,
    "video_selection_strategy": "longest",
    "batch_multiplier": 3
  },
  "audio_settings": {
    "segment_duration_seconds": 30,
    "audio_format": "wav",
    "sample_rate": 16000
  },
  "pipeline_settings": {
    "cleanup_intermediate_files": true,
    "create_dataset_summary": true
  }
}
```

#### Bước 3: Chạy pipeline
```bash
python channel_to_dataset_pipeline.py
```

### Method 2: Interactive Single Channel
```bash
python channel_to_dataset_pipeline.py
# Script sẽ hỏi channel URL và settings
```

### Method 3: Bulk URL Processing

#### Bước 1: Tạo video URL list
```bash
python run_url_crawler.py
# Tạo file youtube_video_urls.txt
```

#### Bước 2: Process URLs
```bash
python youtube_audio_processor.py
```

## Cấu hình chi tiết

### Crawl Settings
- max_videos_per_channel: Số video tối đa/channel (default: 20)
- min_duration_seconds: Thời lượng tối thiểu (default: 1800s = 30 phút)  
- max_duration_seconds: Thời lượng tối đa (default: 7200s = 2 tiếng)
- video_selection_strategy: Chiến lược chọn video
  - "longest": Ưu tiên video dài nhất
  - "shortest": Ưu tiên video ngắn nhất
  - "first": Lấy theo thứ tự upload
- batch_multiplier: Hệ số nhân để lọc (default: 3)

### Audio Settings  
- segment_duration_seconds: Độ dài segment (default: 30s)
- audio_format: Format output (default: "wav")
- sample_rate: Tần số lấy mẫu (default: 16000 Hz)

### Pipeline Settings
- cleanup_intermediate_files: Xóa file tạm (default: true)
- create_dataset_summary: Tạo file tổng kết (default: true)
- parallel_processing: Xử lý song song (experimental)

## Cấu trúc Input/Output

### Input Files
- config.json: File cấu hình chính
- Channel URLs: Link YouTube channels
- youtube_video_urls.txt: Danh sách video URLs (optional)

### Output Structure
```
datasets/
└── {channel_name}/
    ├── dataset/               # Dataset cuối cùng
    │   ├── {video_id}_001.wav    # Audio segments (30s)
    │   ├── {video_id}_001.json   # Transcript labels
    │   ├── {video_id}_002.wav
    │   ├── {video_id}_002.json
    │   └── dataset_summary.json  # Metadata tổng kết
    ├── subtitles/             # Temporary files
    ├── audio/                 # Temporary files  
    └── audio_segments/        # Temporary files
```

### Intermediate Files (Tự động xóa nếu cleanup=true)
- `subtitles/`: File .srt subtitle gốc
- `audio/`: File audio gốc từ YouTube
- `audio_segments/`: File .wav đã chia segments
- `processing_metadata.json`: Metadata xử lý audio
