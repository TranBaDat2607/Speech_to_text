# YouTube Audio And Transcript Crawling Pipeline

English below

Pipeline để tải video YouTube và chuyển đổi thành file audio .wav cho training model speech-to-text.

## Yêu cầu hệ thống

### 1. FFmpeg
```bash
winget install FFmpeg
```

### 2. Python packages
```bash
pip install -r ../requirements.txt
```

## Cách sử dụng

### Phương pháp 1: Sử dụng file cấu hình JSON (Khuyến nghị)

#### Bước 1: Cấu hình
Chỉnh sửa file `config.json` để tùy chỉnh các tham số:

```json
{
  "channels": [
    "https://www.youtube.com/@abc"
  ],
  "crawl_settings": {
    "max_videos_per_channel": 20,
    "min_duration_seconds": 60,
    "max_duration_seconds": 3600,
    "video_selection_strategy": "longest",
    "batch_multiplier": 3
  },
  "subtitle_settings": {
    "languages": ["vi", "en"],
    "format": "srt"
  },
  "audio_settings": {
    "segment_duration_seconds": 30,
    "audio_format": "wav",
    "sample_rate": 16000
  },
  "pipeline_settings": {
    "cleanup_intermediate_files": true,
    "create_dataset_summary": true
  },
  "output_settings": {
    "base_output_dir": "datasets",
    "create_channel_folders": true,
    "save_metadata": true
  }
}
```

#### Bước 2: Chạy pipeline
```bash
python channel_to_dataset_pipeline.py
```

### Phương pháp 2: Chế độ tương tác
```bash
python channel_to_dataset_pipeline.py
# Sau đó gọi main_interactive() thay vì main()
```

### Phương pháp 3: Từ file URLs có sẵn (Cách cũ)
```bash
# Bước 1: Tải subtitle
python download_subtitles_ytdlp.py

# Bước 2: Xử lý audio
python youtube_audio_processor.py

# Bước 3: Tạo dataset labels
python create_dataset_labels.py
```

## Tham số cấu hình quan trọng

### Crawl Settings
- **channels**: Danh sách URL các channel YouTube
- **max_videos_per_channel**: Số video tối đa mỗi channel (mặc định: 20)
- **min_duration_seconds**: Thời lượng video tối thiểu (mặc định: 60s)
- **max_duration_seconds**: Thời lượng video tối đa (mặc định: 3600s)
- **video_selection_strategy**: Chiến lược chọn video
  - `"longest"`: Ưu tiên video dài nhất
  - `"shortest"`: Ưu tiên video ngắn nhất  
  - `"first"`: Lấy theo thứ tự (mặc định)
- **batch_multiplier**: Hệ số nhân để lấy nhiều video hơn rồi lọc (mặc định: 3)

### Audio Settings
- **segment_duration_seconds**: Độ dài mỗi audio segment (mặc định: 30s)
- **audio_format**: Format audio output (mặc định: "wav")
- **sample_rate**: Tần số lấy mẫu (mặc định: 16000 Hz)

### Pipeline Settings
- **cleanup_intermediate_files**: Xóa file trung gian sau khi hoàn thành (mặc định: true)
- **create_channel_folders**: Tạo thư mục riêng cho mỗi channel (mặc định: true)

## Input
- **config.json**: File cấu hình chính (phương pháp 1)
- **Channel URL**: Link YouTube channel (phương pháp 2)
- `youtube_video_urls.txt`: File chứa danh sách URL YouTube (phương pháp 3)

## Output
- `subtitles/`: Thư mục chứa file .srt subtitle
- `audio/`: Thư mục chứa file audio gốc
- `audio_segments/`: Thư mục chứa file .wav 30 giây
- `dataset/`: Thư mục chứa dataset cuối cùng (file .wav + .json)
- `processing_metadata.json`: File metadata xử lý audio
- `dataset_summary.json`: File tổng kết dataset

## Cấu trúc file

### Audio segments
- Format: .wav
- Thời lượng: 30 giây
- Chất lượng: 192kbps
- Tên file: `{video_id}_segment_{số}.wav`

### Dataset labels (JSON)
```json
{
  "start": 0,
  "end": 30,
  "video_id": "ALysMspFXxE",
  "text": "xin chào các bạn tác phẩm bạn đang nghe"
}
```

### Dataset summary
```json
{
  "total_videos": 3,
  "total_labels": 15,
  "videos": {...},
  "created_at": "2025-01-01T10:00:00"
}
```

## Ví dụ sử dụng

### 1. Crawl dataset từ nhiều channel với cấu hình tùy chỉnh
```bash
# Chỉnh sửa config.json với channels và tham số mong muốn
python channel_to_dataset_pipeline.py
```

### 2. Test với một channel đơn lẻ
```python
from channel_to_dataset_pipeline import ChannelToDatasetPipeline
from config_manager import ConfigManager

config = ConfigManager()
pipeline = ChannelToDatasetPipeline("https://www.youtube.com/@abc", config)
pipeline.run_pipeline()
```

### 3. Kiểm tra cấu hình hiện tại
```bash
python config_manager.py
```

## Lưu ý quan trọng

### Về hiệu suất
- **batch_multiplier = 3**: Lấy gấp 3 lần video để đảm bảo đủ sau khi lọc
- **video_selection_strategy = "longest"**: Ưu tiên video dài để có nhiều dữ liệu hơn
- Quá trình crawl có thể mất 10-30 phút tùy số lượng channel

### Về chất lượng dữ liệu
- Chỉ lấy video có subtitle tiếng Việt/tiếng Anh
- Loại bỏ video private, premium, subscriber-only
- Lọc video theo thời lượng để đảm bảo chất lượng

### Về dung lượng
- Mỗi video ~10-50MB tùy độ dài
- Dataset cuối cùng ~100-500MB cho 20 video/channel
- Bật `cleanup_intermediate_files` để tiết kiệm dung lượng

## Xử lý lỗi thường gặp

### 1. "Không crawl được video nào"
- Kiểm tra URL channel có đúng không
- Thử giảm `min_duration_seconds` trong config
- Kiểm tra kết nối internet

### 2. "Video không khả dụng"
- Video có thể bị private/deleted
- Tăng `batch_multiplier` để crawl nhiều video hơn
- Thử channel khác có nhiều video công khai

### 3. "Không tải được subtitle"
- Video có thể không có subtitle
- Thử thêm ngôn ngữ khác vào `subtitle_settings.languages`
- Kiểm tra video có auto-generated captions không

## Cần kết nối internet và file .wav sẵn sàng cho training model

---

# English Documentation

# YouTube Audio And Transcript Crawling Pipeline

Pipeline to download YouTube videos and convert them to .wav audio files for speech-to-text model training.

## System Requirements

### 1. FFmpeg
```bash
winget install FFmpeg
```

### 2. Python packages
```bash
pip install -r ../requirements.txt
```

## Usage

### Method 1: Using JSON Configuration File (Recommended)

#### Step 1: Configuration
Edit the `config.json` file to customize parameters:

```json
{
  "channels": [
    "https://www.youtube.com/@abc"
  ],
  "crawl_settings": {
    "max_videos_per_channel": 10,
    "min_duration_seconds": 1800,
    "max_duration_seconds": 7200,
    "video_selection_strategy": "longest",
    "batch_multiplier": 3
  },
  "subtitle_settings": {
    "languages": ["vi"],
    "format": "srt"
  },
  "audio_settings": {
    "segment_duration_seconds": 30,
    "audio_format": "wav",
    "sample_rate": 16000
  },
  "pipeline_settings": {
    "cleanup_intermediate_files": true,
    "create_dataset_summary": true
  },
  "output_settings": {
    "base_output_dir": "datasets",
    "create_channel_folders": true,
    "save_metadata": true
  }
}
```

#### Step 2: Run Pipeline
```bash
python channel_to_dataset_pipeline.py
```

### Method 2: Interactive Mode
```bash
python channel_to_dataset_pipeline.py
# Then call main_interactive() instead of main()
```

### Method 3: From Existing URLs File (Legacy)
```bash
# Step 1: Download subtitles
python download_subtitles_ytdlp.py

# Step 2: Process audio
python youtube_audio_processor.py

# Step 3: Create dataset labels
python create_dataset_labels.py
```

## Important Configuration Parameters

### Crawl Settings
- **channels**: List of YouTube channel URLs
- **max_videos_per_channel**: Maximum videos per channel (default: 20)
- **min_duration_seconds**: Minimum video duration (default: 60s)
- **max_duration_seconds**: Maximum video duration (default: 3600s)
- **video_selection_strategy**: Video selection strategy
  - `"longest"`: Prioritize longest videos
  - `"shortest"`: Prioritize shortest videos
  - `"first"`: Take in order (default)
- **batch_multiplier**: Multiplier to fetch more videos then filter (default: 3)

### Audio Settings
- **segment_duration_seconds**: Duration of each audio segment (default: 30s)
- **audio_format**: Audio output format (default: "wav")
- **sample_rate**: Sample rate (default: 16000 Hz)

### Pipeline Settings
- **cleanup_intermediate_files**: Delete intermediate files after completion (default: true)
- **create_channel_folders**: Create separate folder for each channel (default: true)

## Input
- **config.json**: Main configuration file (method 1)
- **Channel URL**: YouTube channel link (method 2)
- `youtube_video_urls.txt`: File containing YouTube URL list (method 3)

## Output
- `subtitles/`: Folder containing .srt subtitle files
- `audio/`: Folder containing original audio files
- `audio_segments/`: Folder containing 30-second .wav files
- `dataset/`: Folder containing final dataset (.wav + .json files)
- `processing_metadata.json`: Audio processing metadata file
- `dataset_summary.json`: Dataset summary file

## File Structure

### Audio segments
- Format: .wav
- Duration: 30 seconds
- Quality: 192kbps
- Filename: `{video_id}_segment_{number}.wav`

### Dataset labels (JSON)
```json
{
  "start": 0,
  "end": 30,
  "video_id": "ALysMspFXxE",
  "text": "hello everyone welcome to this video"
}
```

### Dataset summary
```json
{
  "total_videos": 3,
  "total_labels": 15,
  "videos": {...},
  "created_at": "2025-01-01T10:00:00"
}
```

## Usage Examples

### 1. Crawl dataset from multiple channels with custom configuration
```bash
# Edit config.json with desired channels and parameters
python channel_to_dataset_pipeline.py
```

### 2. Test with a single channel
```python
from channel_to_dataset_pipeline import ChannelToDatasetPipeline
from config_manager import ConfigManager

config = ConfigManager()
pipeline = ChannelToDatasetPipeline("https://www.youtube.com/@abc", config)
pipeline.run_pipeline()
```

### 3. Check current configuration
```bash
python config_manager.py
```

## Important Notes

### Performance
- **batch_multiplier = 3**: Fetch 3x videos to ensure enough after filtering
- **video_selection_strategy = "longest"**: Prioritize long videos for more data
- Crawling process may take 10-30 minutes depending on number of channels

### Data Quality
- Only fetch videos with Vietnamese/English subtitles
- Filter out private, premium, subscriber-only videos
- Filter videos by duration to ensure quality

### Storage
- Each video ~10-50MB depending on length
- Final dataset ~100-500MB for 20 videos/channel
- Enable `cleanup_intermediate_files` to save storage

## Common Error Handling

### 1. "Could not crawl any videos"
- Check if channel URL is correct
- Try reducing `min_duration_seconds` in config
- Check internet connection

### 2. "Video not available"
- Videos might be private/deleted
- Increase `batch_multiplier` to crawl more videos
- Try other channels with more public videos

### 3. "Could not download subtitles"
- Videos might not have subtitles
- Try adding other languages to `subtitle_settings.languages`
- Check if videos have auto-generated captions

## Requires internet connection and .wav files ready for model training
