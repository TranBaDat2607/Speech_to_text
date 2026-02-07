# YouTube Audio Dataset Crawler

Automated pipeline for crawling audio and transcripts from YouTube channels to create structured datasets for speech-to-text model training.

## Overview

This pipeline provides an end-to-end solution for creating speech recognition datasets from YouTube content. It handles video crawling, audio extraction, subtitle downloading, audio segmentation, and dataset label generation with built-in error handling and resume capabilities.

Key Philosophy:
- Single workflow: Configure everything in config.json, then run the pipeline
- No defaults: All settings must be explicitly defined
- Fully reproducible: Your config file documents exactly how your dataset was created
- Zero ambiguity: One way to run, no interactive prompts

## Features

Core Functionality:
- Multi-channel crawling with concurrent processing
- Automatic subtitle download (Vietnamese and English with auto-generated fallback)
- Audio segmentation into standardized segments
- Dataset generation with WAV and JSON label pairs
- Smart cleanup of intermediate files
- JSON-based configuration

Resilience:
- Automatic retry logic with exponential backoff (3 attempts per operation)
- Checkpoint and resume support
- Graceful degradation on individual video failures
- File validation for data quality
- Operation tracking with detailed statistics

Monitoring:
- Comprehensive logging to console and pipeline.log
- Dataset validation
- Real-time progress tracking
- Detailed failure reporting

## System Requirements

Python Dependencies:
```bash
pip install yt-dlp pydub requests beautifulsoup4 pathlib
```

External Dependencies:

FFmpeg (Required for audio processing):
- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- macOS: brew install ffmpeg
- Ubuntu/Debian: sudo apt install ffmpeg

Verify installation:
```bash
ffmpeg -version
```

## Quick Start

Step 1: Create Configuration File
```bash
cp config.json.example config.json
```

Step 2: Edit Configuration

Edit config.json with your settings:
```json
{
  "channels": [
    "https://www.youtube.com/@channel1",
    "https://www.youtube.com/@channel2"
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
    "save_metadata": true,
    "subtitles_folder": "subtitles",
    "audio_folder": "audio",
    "audio_segments_folder": "audio_segments",
    "dataset_folder": "dataset"
  }
}
```

Step 3: Run Pipeline
```bash
python channel_to_dataset_pipeline.py
```

The pipeline automatically:
1. Crawls videos from all channels in config.json
2. Downloads subtitles for each video
3. Downloads and segments audio files
4. Generates dataset labels (audio + JSON pairs)
5. Validates dataset integrity
6. Cleans up intermediate files (if enabled)

Resuming Failed Runs:

If the pipeline fails or is interrupted, rerun the same command:
```bash
python channel_to_dataset_pipeline.py
```

The pipeline automatically resumes from the last successful step using checkpoint files.

## Configuration Reference

All configuration settings are required. The pipeline will fail with a clear error message if any setting is missing.

Crawl Settings:

- max_videos_per_channel: Maximum number of videos to process per channel
- min_duration_seconds: Minimum video duration in seconds
- max_duration_seconds: Maximum video duration in seconds
- video_selection_strategy: Strategy for selecting videos ("longest", "shortest", "first")
- batch_multiplier: Multiplier for initial video fetch to ensure enough videos after filtering

Subtitle Settings:

- languages: List of preferred subtitle languages in priority order (e.g., ["vi", "en"])
- format: Subtitle format to download ("srt", "vtt", "ass")

Audio Settings:

- segment_duration_seconds: Duration of each audio segment in seconds
- audio_format: Output audio format ("wav", "mp3", "flac")
- sample_rate: Audio sample rate in Hz (16000 recommended for speech recognition)

Pipeline Settings:

- cleanup_intermediate_files: Whether to delete intermediate files after completion
- create_dataset_summary: Whether to generate dataset_summary.json with statistics

Output Settings:

- base_output_dir: Root directory for all generated datasets
- create_channel_folders: Whether to create separate folders for each channel
- save_metadata: Whether to save detailed metadata files
- subtitles_folder: Folder name for downloaded subtitle files
- audio_folder: Folder name for downloaded full-length audio files
- audio_segments_folder: Folder name for segmented audio files
- dataset_folder: Folder name for final dataset output

## Pipeline Architecture

Processing Steps:

1. Channel Crawling
   - Fetches video list from YouTube channel
   - Filters videos by duration constraints
   - Applies selection strategy
   - Generates youtube_video_urls.txt

2. Subtitle Download
   - Downloads subtitles for each video
   - Validates subtitle file integrity
   - Creates .srt and .txt files
   - Continues on failure for individual videos

3. Audio Processing
   - Downloads audio from YouTube
   - Validates audio file integrity
   - Segments audio into fixed-duration chunks
   - Exports as WAV files

4. Label Generation
   - Matches subtitle timestamps with audio segments
   - Creates JSON label files with transcripts
   - Validates dataset pairs (WAV + JSON)

5. Dataset Validation
   - Checks for missing or orphaned files
   - Validates file formats and content
   - Generates validation report

6. Cleanup
   - Removes intermediate files (if enabled)
   - Keeps only final dataset folder

Checkpoint System:

- Progress is saved after each successful step
- If pipeline fails or is interrupted, rerun the same command
- Pipeline automatically resumes from the last successful step
- Prevents reprocessing already completed work
- Checkpoint files are automatically cleaned on successful completion

Error Handling and Retry Logic:

All network operations implement automatic retry with exponential backoff:
- Retry attempts: 3 attempts per operation
- Initial delay: 2-5 seconds
- Backoff multiplier: 2x (delay doubles after each failure)
- Graceful degradation: Pipeline continues even if individual videos fail

## Directory Structure

Input Files:
```
crawl_data/
├── config.json                      # Main configuration file
├── config.json.example              # Example configuration template
├── youtube_video_urls.txt           # Video URL list (optional)
└── *.py                             # Pipeline modules
```

Output Structure:
```
datasets/
└── {channel_name}/
    ├── dataset/                     # Final dataset (keep this)
    │   ├── {video_id}_001.wav       # Audio segment 1 (30 seconds)
    │   ├── {video_id}_001.json      # Transcript label for segment 1
    │   ├── {video_id}_002.wav       # Audio segment 2
    │   ├── {video_id}_002.json      # Transcript label for segment 2
    │   └── dataset_summary.json     # Dataset metadata and statistics
    │
    ├── subtitles/                   # Intermediate (auto-deleted if cleanup=true)
    │   ├── {video_id}.vi.srt        # Vietnamese subtitle
    │   └── {video_id}.en.srt        # English subtitle
    │
    ├── audio/                       # Intermediate (auto-deleted if cleanup=true)
    │   └── {video_id}.wav           # Full-length audio
    │
    └── audio_segments/              # Intermediate (auto-deleted if cleanup=true)
        ├── {video_id}_segment_001.wav
        └── {video_id}_segment_002.wav
```

Generated Files:

Dataset Files (Final output - never deleted):
- {video_id}_{segment_number}.wav: Audio segments for training
- {video_id}_{segment_number}.json: Corresponding transcript labels
- dataset_summary.json: Dataset statistics and metadata

Checkpoint Files (Temporary):
- pipeline_{channel_name}_checkpoint.json: Resume checkpoint data
- Automatically deleted on successful completion

Log Files:
- pipeline.log: Complete execution log with timestamps and errors

## Dataset Label Format

Each JSON label file contains:
```json
{
  "video_id": "VIDEO_ID",
  "start": 0.0,
  "end": 30.0,
  "text": "Transcript text for this audio segment"
}
```

Fields:
- video_id: YouTube video identifier
- start: Start time in seconds (relative to original video)
- end: End time in seconds (relative to original video)
- text: Transcript text synchronized with the audio segment

## Monitoring and Logs

Console Output:

Real-time progress information displayed during execution:
- Current step being executed
- Video processing progress
- Success/failure counts
- Validation results

Log Files:

pipeline.log contains:
- Timestamp for each operation
- INFO: Normal operations and progress
- WARNING: Non-critical issues (e.g., skipped videos)
- ERROR: Critical failures with full stack traces
- Summary statistics at completion

Operation Statistics:

At the end of each step, detailed statistics are displayed:
```
Audio Processing Summary:
  Total: 20
  Successful: 18
  Failed: 2
  Success Rate: 90.0%
  Failed items: 2
```

## Troubleshooting

Common Issues:

Issue: "FFmpeg not found"
- Solution: Install FFmpeg and ensure it's in system PATH
- Verify: Run ffmpeg -version in terminal

Issue: "No subtitles found for video"
- Solution: This is expected for some videos. Pipeline continues with remaining videos
- Check: Review failed videos list in operation summary

Issue: "yt-dlp error: Video unavailable"
- Solution: Video may be private, deleted, or region-restricted
- Action: Pipeline automatically skips and continues

Issue: "Out of disk space"
- Solution: Enable cleanup_intermediate_files in config.json
- Alternative: Reduce max_videos_per_channel or process fewer channels

Issue: "Pipeline slow on large channels"
- Solution: Reduce max_videos_per_channel or adjust batch_multiplier
- Alternative: Use video_selection_strategy="first" for faster processing

Resume Failed Pipeline:

If pipeline fails or is interrupted:
1. Check pipeline.log for error details
2. Fix any configuration issues in config.json if needed
3. Rerun the same command: python channel_to_dataset_pipeline.py
4. Pipeline automatically resumes from last successful step

To start fresh (ignore checkpoint):
```bash
# Remove checkpoint files
rm pipeline_*_checkpoint.json
# Run pipeline
python channel_to_dataset_pipeline.py
```

Validate Dataset:

To validate an existing dataset:
```python
from file_validators import DatasetValidator
from pathlib import Path

validator = DatasetValidator(Path("datasets/channel_name/dataset"))
results = validator.validate_dataset()
validator.print_validation_report()
```

## Performance Optimization

Disk Space Management:

Estimated disk space usage per video:
- Full audio: 10-50 MB per video
- Audio segments: 10-50 MB per video (total)
- Subtitles: 100-500 KB per video
- Final dataset: 10-50 MB per video

Recommendation: Enable cleanup_intermediate_files: true to reduce disk usage by 2-3x.

Processing Time:

Typical processing times (per video):
- Channel crawling: 1-2 seconds per video
- Subtitle download: 5-10 seconds per video
- Audio download: 20-60 seconds per video
- Audio segmentation: 10-30 seconds per video
- Label generation: 1-2 seconds per video

Example: Processing 100 videos (30 minutes average) takes approximately 1-2 hours.

## Best Practices

Configuration Recommendations:

For high-quality datasets:
```json
{
  "crawl_settings": {
    "min_duration_seconds": 300,
    "video_selection_strategy": "longest"
  },
  "audio_settings": {
    "sample_rate": 16000,
    "segment_duration_seconds": 30
  }
}
```

For quick testing:
```json
{
  "crawl_settings": {
    "max_videos_per_channel": 5,
    "min_duration_seconds": 60,
    "max_duration_seconds": 600
  }
}
```

For production datasets:
```json
{
  "crawl_settings": {
    "max_videos_per_channel": 50,
    "min_duration_seconds": 300,
    "max_duration_seconds": 1800
  },
  "pipeline_settings": {
    "cleanup_intermediate_files": true
  }
}
```

Dataset Quality Tips:

1. Choose appropriate channels: Select channels with clear speech and minimal background noise
2. Filter by duration: Use min_duration_seconds >= 300 for better subtitle quality
3. Validate output: Review validation report after pipeline completion
4. Check failed videos: Investigate high failure rates (>20%)
5. Manual review: Sample random segments to verify quality

Maintenance:

Regular cleanup:
```bash
# Remove old checkpoints
rm *_checkpoint.json

# Remove old logs (keep recent ones)
find . -name "pipeline.log" -mtime +30 -delete
```

Dataset versioning:
```bash
# Archive completed datasets
tar -czf dataset_v1_$(date +%Y%m%d).tar.gz datasets/

# Upload to storage
# aws s3 cp dataset_v1_*.tar.gz s3://your-bucket/
```

## Module Reference

Core Modules:

- channel_to_dataset_pipeline.py: Main pipeline orchestrator
- config_manager.py: Configuration file handling
- single_channel_crawler.py: YouTube channel video crawling
- youtube_audio_processor.py: Audio download and segmentation
- download_subtitles_ytdlp.py: Subtitle download via yt-dlp
- create_dataset_labels.py: Dataset label generation

Utility Modules:

- retry_utils.py: Retry decorators and operation tracking
- checkpoint_manager.py: Pipeline checkpoint and resume logic
- file_validators.py: File integrity validation

Helper Scripts:

- run_url_crawler.py: Standalone URL crawler
- youtube_channel_crawler.py: Multi-channel URL extraction

## Advanced Usage

Using Custom Config File:

```python
from config_manager import ConfigManager
from channel_to_dataset_pipeline import ChannelToDatasetPipeline

# Load custom config
config = ConfigManager("my_custom_config.json")

# Get channels from config
channels = config.get_channels()

# Process each channel
for channel_url in channels:
    pipeline = ChannelToDatasetPipeline(channel_url, config, enable_checkpoint=True)
    success = pipeline.run_pipeline()

    if not success:
        print(f"Failed: {channel_url}")
```

Running Individual Pipeline Steps:

For debugging or custom workflows, you can run individual steps:
```python
from channel_to_dataset_pipeline import ChannelToDatasetPipeline
from config_manager import ConfigManager

# Load config (required)
config = ConfigManager("config.json")

# Create pipeline instance
pipeline = ChannelToDatasetPipeline(
    channel_url="https://www.youtube.com/@channel",
    config=config,
    enable_checkpoint=True
)

# Run steps individually
pipeline.step1_crawl_channel()
pipeline.step2_download_subtitles()
pipeline.step3_process_audio()
pipeline.step4_create_labels()
pipeline.step5_validate_dataset()
pipeline.step6_cleanup_intermediate_files()
```

Note: All steps require a valid config.json file with all required settings. There are no default values.
