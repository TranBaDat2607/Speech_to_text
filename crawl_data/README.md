# YouTube Audio Dataset Crawler

A production-ready automated pipeline for crawling audio and transcripts from YouTube channels to create structured datasets for speech-to-text machine learning model training.

## Overview

This pipeline provides an end-to-end solution for creating high-quality speech recognition datasets from YouTube content. It handles video crawling, audio extraction, subtitle downloading, audio segmentation, and dataset label generation with built-in error handling, retry mechanisms, and resume capabilities.

**Key Philosophy:**
- **Single workflow:** Configure everything in `config.json`, then run the pipeline
- **No defaults:** All settings must be explicitly defined in config - fail fast with clear errors
- **Fully reproducible:** Your config file documents exactly how your dataset was created
- **Zero ambiguity:** One way to run, no interactive prompts, no hidden behaviors

## Key Features

### Core Functionality
- **Multi-channel crawling**: Process multiple YouTube channels concurrently
- **Automatic subtitle download**: Downloads Vietnamese and English subtitles with fallback to auto-generated captions
- **Audio segmentation**: Splits audio into standardized segments (default: 30 seconds)
- **Dataset generation**: Creates structured WAV and JSON label pairs ready for ML training
- **Smart cleanup**: Automatically removes intermediate files to conserve disk space
- **JSON configuration**: Flexible configuration through JSON files

### Resilience and Error Handling
- **Automatic retry logic**: Network failures are automatically retried with exponential backoff (3 attempts per operation)
- **Checkpoint and resume**: Pipeline saves progress and can resume from last successful step if interrupted
- **Graceful degradation**: Continues processing remaining items even if individual videos fail
- **File validation**: Validates audio and subtitle files to ensure data quality
- **Operation tracking**: Provides detailed success/failure statistics for each operation

### Monitoring and Logging
- **Comprehensive logging**: All operations logged to both console and file (pipeline.log)
- **Dataset validation**: Automatic validation of generated dataset integrity
- **Progress tracking**: Real-time progress updates during processing
- **Failure reporting**: Detailed reports of failed operations with error messages

## System Requirements

### Python Dependencies

Install required Python packages:

```bash
pip install yt-dlp pydub requests beautifulsoup4 pathlib
```

### External Dependencies

**FFmpeg** (Required for audio processing):
- Windows: Download from https://ffmpeg.org/download.html and add to PATH
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt install ffmpeg`

Verify FFmpeg installation:
```bash
ffmpeg -version
```

## Quick Start

The pipeline uses a single, simple workflow: configure everything in `config.json`, then run the pipeline.

### Step 1: Create Configuration File

```bash
cp config.json.example config.json
```

### Step 2: Edit Configuration

Edit `config.json` with your desired settings:

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

### Step 3: Run Pipeline

```bash
python channel_to_dataset_pipeline.py
```

**That's it!** The pipeline will automatically:
1. Crawl videos from all channels in config.json
2. Download subtitles for each video
3. Download and segment audio files
4. Generate dataset labels (audio + JSON pairs)
5. Validate dataset integrity
6. Clean up intermediate files (if enabled)

### Resuming Failed Runs

If the pipeline fails or is interrupted, simply rerun the same command:

```bash
python channel_to_dataset_pipeline.py
```

The pipeline automatically resumes from the last successful step using checkpoint files.

## Configuration Reference

**Important:** All configuration settings are **required**. The pipeline will fail with a clear error message if any setting is missing. There are no default values.

### Crawl Settings

Configuration for video selection and filtering.

- **max_videos_per_channel** (integer, required)
  - Maximum number of videos to process per channel
  - Higher values create larger datasets but take longer to process
  - Example: 20

- **min_duration_seconds** (integer, required)
  - Minimum video duration in seconds
  - Videos shorter than this are skipped
  - Recommended: 300 seconds (5 minutes) for better transcript quality
  - Example: 60

- **max_duration_seconds** (integer, required)
  - Maximum video duration in seconds
  - Videos longer than this are skipped
  - Recommended: 1800 seconds (30 minutes) for manageable file sizes
  - Example: 3600

- **video_selection_strategy** (string, required)
  - Strategy for selecting videos when more than max_videos are available
  - Options:
    - "longest": Prioritizes longest videos (best for maximizing dataset duration)
    - "shortest": Prioritizes shortest videos
    - "first": Takes videos in upload order
  - Example: "longest"

- **batch_multiplier** (integer, required)
  - Multiplier for initial video fetch to ensure enough videos after filtering
  - Example: If max_videos=20 and batch_multiplier=3, fetches 60 videos then filters to 20
  - Increase if many videos are filtered out by duration constraints
  - Example: 3

### Subtitle Settings

Configuration for subtitle download.

- **languages** (array, required)
  - List of preferred subtitle languages in priority order
  - Falls back to auto-generated captions if manual subtitles unavailable
  - Supported: "vi", "en", "zh", "ja", "ko", "fr", "de", "es"
  - Example: ["vi", "en"]

- **format** (string, required)
  - Subtitle format to download
  - Options: "srt", "vtt", "ass"
  - Example: "srt"

### Audio Settings

Configuration for audio processing and segmentation.

- **segment_duration_seconds** (integer, required)
  - Duration of each audio segment in seconds
  - Common values: 10, 20, 30 seconds depending on model requirements
  - Example: 30

- **audio_format** (string, required)
  - Output audio format
  - Options: "wav" (recommended for ML), "mp3", "flac"
  - Example: "wav"

- **sample_rate** (integer, required)
  - Audio sample rate in Hz
  - Common values:
    - 16000 Hz: Standard for speech recognition (recommended)
    - 8000 Hz: Lower quality, smaller files
    - 22050 Hz: Higher quality
    - 44100 Hz: CD quality (unnecessary for speech)
  - Example: 16000

### Pipeline Settings

Configuration for pipeline behavior.

- **cleanup_intermediate_files** (boolean, required)
  - Whether to delete intermediate files after successful completion
  - When true: Saves disk space by removing audio/, audio_segments/, subtitles/ folders
  - When false: Keeps all intermediate files for debugging or reprocessing
  - Example: true

- **create_dataset_summary** (boolean, required)
  - Whether to generate dataset_summary.json with statistics
  - Includes video counts, duration, segment counts, etc.
  - Example: true

### Output Settings

Configuration for output directory structure.

- **base_output_dir** (string, required)
  - Root directory for all generated datasets
  - Example: "datasets"

- **create_channel_folders** (boolean, required)
  - Whether to create separate folders for each channel
  - When true: Organizes datasets by channel name
  - Example: true

- **save_metadata** (boolean, required)
  - Whether to save detailed metadata files
  - Example: true

- **subtitles_folder** (string, required)
  - Folder name for downloaded subtitle files
  - Example: "subtitles"

- **audio_folder** (string, required)
  - Folder name for downloaded full-length audio files
  - Example: "audio"

- **audio_segments_folder** (string, required)
  - Folder name for segmented audio files
  - Example: "audio_segments"

- **dataset_folder** (string, required)
  - Folder name for final dataset output (audio + JSON pairs)
  - Example: "dataset"

## Pipeline Architecture

### Processing Steps

The pipeline executes the following steps sequentially:

1. **Channel Crawling**
   - Fetches video list from YouTube channel
   - Filters videos by duration constraints
   - Applies selection strategy
   - Generates youtube_video_urls.txt

2. **Subtitle Download**
   - Downloads subtitles for each video
   - Validates subtitle file integrity
   - Creates .srt and .txt files
   - Continues on failure for individual videos

3. **Audio Processing**
   - Downloads audio from YouTube
   - Validates audio file integrity
   - Segments audio into fixed-duration chunks
   - Exports as WAV files

4. **Label Generation**
   - Matches subtitle timestamps with audio segments
   - Creates JSON label files with transcripts
   - Validates dataset pairs (WAV + JSON)

5. **Dataset Validation**
   - Checks for missing or orphaned files
   - Validates file formats and content
   - Generates validation report

6. **Cleanup**
   - Removes intermediate files (if enabled)
   - Keeps only final dataset folder

### Checkpoint System

The pipeline implements an automatic checkpoint system:

- Progress is saved after each successful step
- If pipeline fails or is interrupted, rerun the same command
- Pipeline automatically resumes from the last successful step
- Prevents reprocessing already completed work
- Checkpoint files are automatically cleaned on successful completion

Example checkpoint recovery:

```bash
# First run - fails at Step 3
$ python channel_to_dataset_pipeline.py
# Output: "Pipeline failed at Step 3. Checkpoint saved for resume."

# Second run - automatically resumes
$ python channel_to_dataset_pipeline.py
# Output: "Resuming from checkpoint: Steps 1-2 completed, starting Step 3..."
```

### Error Handling and Retry Logic

All network operations implement automatic retry with exponential backoff:

- **Retry attempts**: 3 attempts per operation
- **Initial delay**: 2-5 seconds (varies by operation)
- **Backoff multiplier**: 2x (delay doubles after each failure)
- **Graceful degradation**: Pipeline continues even if individual videos fail

Operation tracking provides detailed statistics:
- Total attempts
- Successful operations
- Failed operations
- Success rate percentage
- List of failed items with error messages

## Directory Structure

### Input Files

```
crawl_data/
├── config.json                      # Main configuration file
├── config.json.example              # Example configuration template
├── youtube_video_urls.txt           # Video URL list (optional)
└── *.py                             # Pipeline modules
```

### Output Structure

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

### Generated Files

**Dataset Files** (Final output - never deleted):
- `{video_id}_{segment_number}.wav`: Audio segments for training
- `{video_id}_{segment_number}.json`: Corresponding transcript labels
- `dataset_summary.json`: Dataset statistics and metadata

**Checkpoint Files** (Temporary):
- `pipeline_{channel_name}_checkpoint.json`: Resume checkpoint data
- Automatically deleted on successful completion

**Log Files**:
- `pipeline.log`: Complete execution log with timestamps and errors

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
- **video_id**: YouTube video identifier
- **start**: Start time in seconds (relative to original video)
- **end**: End time in seconds (relative to original video)
- **text**: Transcript text synchronized with the audio segment

## Monitoring and Logs

### Console Output

Real-time progress information displayed during execution:
- Current step being executed
- Video processing progress (X/Y videos)
- Success/failure counts
- Validation results

### Log Files

**pipeline.log**: Complete execution log including:
- Timestamp for each operation
- INFO: Normal operations and progress
- WARNING: Non-critical issues (e.g., skipped videos)
- ERROR: Critical failures with full stack traces
- Summary statistics at completion

### Operation Statistics

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

### Common Issues

**Issue**: "FFmpeg not found"
- **Solution**: Install FFmpeg and ensure it's in system PATH
- **Verify**: Run `ffmpeg -version` in terminal

**Issue**: "No subtitles found for video"
- **Solution**: This is expected for some videos. Pipeline continues with remaining videos
- **Check**: Review failed videos list in operation summary

**Issue**: "yt-dlp error: Video unavailable"
- **Solution**: Video may be private, deleted, or region-restricted
- **Action**: Pipeline automatically skips and continues

**Issue**: "Out of disk space"
- **Solution**: Enable cleanup_intermediate_files in config.json
- **Alternative**: Reduce max_videos_per_channel or process fewer channels

**Issue**: "Pipeline slow on large channels"
- **Solution**: Reduce max_videos_per_channel or adjust batch_multiplier
- **Alternative**: Use video_selection_strategy="first" for faster processing

### Resume Failed Pipeline

If pipeline fails or is interrupted:

1. Check `pipeline.log` for error details
2. Fix any configuration issues in `config.json` if needed
3. Rerun the same command:
   ```bash
   python channel_to_dataset_pipeline.py
   ```
4. Pipeline automatically resumes from last successful step

To start fresh (ignore checkpoint):
```bash
# Remove checkpoint files
rm pipeline_*_checkpoint.json
# Run pipeline
python channel_to_dataset_pipeline.py
```

### Validate Dataset

To validate an existing dataset:

```python
from file_validators import DatasetValidator
from pathlib import Path

validator = DatasetValidator(Path("datasets/channel_name/dataset"))
results = validator.validate_dataset()
validator.print_validation_report()
```

## Performance Optimization

### Disk Space Management

Estimated disk space usage per video:
- Full audio: ~10-50 MB per video
- Audio segments: ~10-50 MB per video (total)
- Subtitles: ~100-500 KB per video
- Final dataset: ~10-50 MB per video

**Recommendation**: Enable `cleanup_intermediate_files: true` to reduce disk usage by 2-3x.

### Processing Time

Typical processing times (per video):
- Channel crawling: 1-2 seconds per video
- Subtitle download: 5-10 seconds per video
- Audio download: 20-60 seconds per video (depends on video length)
- Audio segmentation: 10-30 seconds per video
- Label generation: 1-2 seconds per video

**Example**: Processing 100 videos (30 minutes average) takes approximately 1-2 hours.


## Best Practices

### Configuration Recommendations

**For high-quality datasets**:
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

**For quick testing**:
```json
{
  "crawl_settings": {
    "max_videos_per_channel": 5,
    "min_duration_seconds": 60,
    "max_duration_seconds": 600
  }
}
```

**For production datasets**:
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

### Dataset Quality Tips

1. **Choose appropriate channels**: Select channels with clear speech and minimal background noise
2. **Filter by duration**: Use min_duration_seconds >= 300 for better subtitle quality
3. **Validate output**: Review validation report after pipeline completion
4. **Check failed videos**: Investigate high failure rates (>20%)
5. **Manual review**: Sample random segments to verify quality

### Maintenance

**Regular cleanup**:
```bash
# Remove old checkpoints
rm *_checkpoint.json

# Remove old logs (keep recent ones)
find . -name "pipeline.log" -mtime +30 -delete
```

**Dataset versioning**:
```bash
# Archive completed datasets
tar -czf dataset_v1_$(date +%Y%m%d).tar.gz datasets/

# Upload to storage
# aws s3 cp dataset_v1_*.tar.gz s3://your-bucket/
```

## Module Reference

### Core Modules

- **channel_to_dataset_pipeline.py**: Main pipeline orchestrator
- **config_manager.py**: Configuration file handling
- **single_channel_crawler.py**: YouTube channel video crawling
- **youtube_audio_processor.py**: Audio download and segmentation
- **download_subtitles_ytdlp.py**: Subtitle download via yt-dlp
- **create_dataset_labels.py**: Dataset label generation

### Utility Modules

- **retry_utils.py**: Retry decorators and operation tracking
- **checkpoint_manager.py**: Pipeline checkpoint and resume logic
- **file_validators.py**: File integrity validation

### Helper Scripts

- **run_url_crawler.py**: Standalone URL crawler
- **youtube_channel_crawler.py**: Multi-channel URL extraction

## Advanced Usage

### Using Custom Config File

To use a different configuration file:

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

### Running Individual Pipeline Steps

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

**Note:** All steps require a valid `config.json` file with all required settings. There are no default values.
