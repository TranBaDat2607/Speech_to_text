# YouTube Audio Processing Pipeline

Pipeline để tải video YouTube và chuyển đổi thành file audio .wav 30 giây cho training model.

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

### Phương pháp 1: Từ channel URL (Khuyến nghị)
```bash
python channel_to_dataset_pipeline.py
```
Pipeline sẽ hỏi channel URL và số video muốn xử lý.

### Phương pháp 2: Từ file URLs có sẵn
```bash
# Bước 1: Tải subtitle
python download_subtitles_ytdlp.py

# Bước 2: Xử lý audio
python youtube_audio_processor.py

# Bước 3: Tạo dataset labels
python create_dataset_labels.py
```


## Input
- **Channel URL**: Link YouTube channel (phương pháp 1)
- `youtube_video_urls.txt`: File chứa danh sách URL YouTube (phương pháp 2)

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

## Lưu ý
- Cần kết nối internet
- Quá trình có thể mất vài phút
- File .wav sẵn sàng cho training model
