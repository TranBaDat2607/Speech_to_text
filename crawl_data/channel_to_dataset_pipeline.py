#!/usr/bin/env python3
"""
Channel to Dataset Pipeline
Pipeline hoàn chỉnh từ YouTube channel URL đến dataset speech-to-text
"""

import os
import sys
import time
import shutil
from pathlib import Path
import logging
from single_channel_crawler import SingleChannelCrawler
from download_subtitles_ytdlp import YouTubeSubtitleDownloader
from youtube_audio_processor import YouTubeAudioProcessor
from create_dataset_labels import DatasetLabeler
from config_manager import ConfigManager
from checkpoint_manager import PipelineCheckpoint
from file_validators import DatasetValidator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ChannelToDatasetPipeline:
    def __init__(self, channel_url: str, config: ConfigManager = None, enable_checkpoint: bool = True):
        self.channel_url = channel_url
        self.config = config if config else ConfigManager()
        self.max_videos = self.config.get_max_videos_per_channel()
        self.cleanup = self.config.get_cleanup_mode()
        self.results = {}
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint = PipelineCheckpoint(f"pipeline_{self._get_channel_name()}") if enable_checkpoint else None
        logger.info(f"Initialized pipeline for channel: {channel_url}")

    def _get_channel_name(self):
        """Extract channel name from URL for checkpoint naming"""
        if '@' in self.channel_url:
            return self.channel_url.split('@')[-1].split('/')[0]
        return "unknown_channel"
        
    def step1_crawl_channel(self):
        """Step 1: Crawl channel to get video URLs"""
        step_name = "crawl_channel"

        # Check if step already completed
        if self.checkpoint and self.checkpoint.is_step_complete(step_name):
            logger.info(f"Step '{step_name}' already completed, skipping...")
            self.results['crawl'] = self.checkpoint.step_results.get(step_name)
            return True

        logger.info("\n" + "="*60)
        logger.info("STEP 1: CRAWL CHANNEL VIDEOS")
        logger.info("="*60)

        try:
            crawler = SingleChannelCrawler()

            # Crawl channel with config settings
            result = crawler.crawl_channel(
                channel_url=self.channel_url,
                max_videos=self.max_videos,
                filter_duration=False,
                min_duration=self.config.get_min_duration(),
                max_duration=self.config.get_max_duration(),
                selection_strategy=self.config.get_video_selection_strategy(),
                batch_multiplier=self.config.get_batch_multiplier()
            )

            if not result:
                logger.error("Could not crawl any videos")
                return False

            self.results['crawl'] = result
            logger.info(f"Crawled {result['total_videos']} videos from channel")

            # Save checkpoint
            if self.checkpoint:
                self.checkpoint.mark_step_complete(step_name, result)

            return True

        except Exception as e:
            logger.error(f"Error in step1_crawl_channel: {e}", exc_info=True)
            return False
        
    def step2_download_subtitles(self):
        """Step 2: Download subtitles"""
        step_name = "download_subtitles"

        # Check if step already completed
        if self.checkpoint and self.checkpoint.is_step_complete(step_name):
            logger.info(f"Step '{step_name}' already completed, skipping...")
            self.results['subtitles'] = self.checkpoint.step_results.get(step_name, 0)
            return True

        logger.info("\n" + "="*60)
        logger.info("STEP 2: DOWNLOAD SUBTITLES")
        logger.info("="*60)

        try:
            if not os.path.exists("youtube_video_urls.txt"):
                logger.error("Could not find file youtube_video_urls.txt")
                return False

            downloader = YouTubeSubtitleDownloader(
                download_folder=self.config.get_subtitles_folder(),
                languages=self.config.get_subtitle_languages(),
                subtitle_format=self.config.get_subtitle_format()
            )
            downloader.process_url_file("youtube_video_urls.txt")

            # Check results
            subtitles_folder = Path(self.config.get_subtitles_folder())
            if subtitles_folder.exists():
                srt_files = list(subtitles_folder.glob("*.srt"))
                logger.info(f"Downloaded {len(srt_files)} subtitle files")
                self.results['subtitles'] = len(srt_files)

                # Save checkpoint
                if self.checkpoint:
                    self.checkpoint.mark_step_complete(step_name, len(srt_files))

                return len(srt_files) > 0

            logger.warning("Subtitles folder not found")
            return False

        except Exception as e:
            logger.error(f"Error in step2_download_subtitles: {e}", exc_info=True)
            return False
        
    def step3_process_audio(self):
        """Step 3: Process audio"""
        step_name = "process_audio"

        # Check if step already completed
        if self.checkpoint and self.checkpoint.is_step_complete(step_name):
            logger.info(f"Step '{step_name}' already completed, skipping...")
            self.results['audio_segments'] = self.checkpoint.step_results.get(step_name, 0)
            return True

        logger.info("\n" + "="*60)
        logger.info("STEP 3: PROCESS AUDIO")
        logger.info("="*60)

        try:
            processor = YouTubeAudioProcessor(
                audio_folder=self.config.get_audio_folder(),
                output_folder=self.config.get_audio_segments_folder(),
                segment_duration=self.config.get_segment_duration(),
                audio_format=self.config.get_audio_format()
            )
            results = processor.process_urls_from_file("youtube_video_urls.txt", limit=self.max_videos)

            # Check results
            segments_folder = Path(self.config.get_audio_segments_folder())
            if segments_folder.exists():
                audio_pattern = f"*.{self.config.get_audio_format()}"
                audio_files = list(segments_folder.glob(audio_pattern))
                logger.info(f"Created {len(audio_files)} audio segment files")
                self.results['audio_segments'] = len(audio_files)

                # Save checkpoint
                if self.checkpoint:
                    self.checkpoint.mark_step_complete(step_name, len(audio_files))

                return len(audio_files) > 0

            logger.warning("Audio segments folder not found")
            return False

        except Exception as e:
            logger.error(f"Error in step3_process_audio: {e}", exc_info=True)
            return False
        
    def step4_create_labels(self):
        """Step 4: Create dataset labels"""
        step_name = "create_labels"

        # Check if step already completed
        if self.checkpoint and self.checkpoint.is_step_complete(step_name):
            logger.info(f"Step '{step_name}' already completed, skipping...")
            self.results['dataset'] = self.checkpoint.step_results.get(step_name, {})
            return True

        logger.info("\n" + "="*60)
        logger.info("STEP 4: CREATE DATASET LABELS")
        logger.info("="*60)

        try:
            labeler = DatasetLabeler(
                subtitles_folder=self.config.get_subtitles_folder(),
                segments_folder=self.config.get_audio_segments_folder(),
                output_folder=self.config.get_dataset_folder(),
                segment_duration=self.config.get_segment_duration(),
                audio_format=self.config.get_audio_format()
            )
            results = labeler.process_all_videos()

            # Check results
            dataset_folder = Path(self.config.get_dataset_folder())
            if dataset_folder.exists():
                audio_pattern = f"*.{self.config.get_audio_format()}"
                audio_files = list(dataset_folder.glob(audio_pattern))
                json_files = list(dataset_folder.glob("*.json"))
                logger.info(f"Created {len(audio_files)} audio files and {len(json_files)} .json files")

                dataset_result = {
                    'audio_files': len(audio_files),
                    'json_files': len(json_files)
                }
                self.results['dataset'] = dataset_result

                # Save checkpoint
                if self.checkpoint:
                    self.checkpoint.mark_step_complete(step_name, dataset_result)

                return len(audio_files) > 0 and len(json_files) > 0

            logger.warning("Dataset folder not found")
            return False

        except Exception as e:
            logger.error(f"Error in step4_create_labels: {e}", exc_info=True)
            return False

    def step5_validate_dataset(self):
        """Step 5: Validate dataset integrity"""
        step_name = "validate_dataset"

        logger.info("\n" + "="*60)
        logger.info("STEP 5: VALIDATE DATASET")
        logger.info("="*60)

        try:
            dataset_folder = Path(self.config.get_dataset_folder())
            if not dataset_folder.exists():
                logger.error("Dataset folder does not exist")
                return True  # Not critical, continue

            validator = DatasetValidator(dataset_folder)
            validation_results = validator.validate_dataset()
            validator.print_validation_report()

            self.results['validation'] = validation_results

            # Save checkpoint
            if self.checkpoint:
                self.checkpoint.mark_step_complete(step_name, validation_results)

            return True

        except Exception as e:
            logger.error(f"Error in step5_validate_dataset: {e}", exc_info=True)
            return True  # Validation failure shouldn't stop the pipeline
        
    def step6_cleanup_intermediate_files(self):
        """Step 6: Cleanup intermediate files to save space"""
        if not self.cleanup:
            logger.info("Skipping cleanup - keeping all files")
            return True

        logger.info("\n" + "="*60)
        logger.info("STEP 6: CLEANUP - DELETE INTERMEDIATE FILES")
        logger.info("="*60)

        folders_to_remove = [
            self.config.get_audio_folder(),
            self.config.get_audio_segments_folder(),
            self.config.get_subtitles_folder()
        ]
        files_to_remove = ["youtube_video_urls.txt", "youtube_video_urls.json", "processing_metadata.json"]
        
        total_size_before = 0
        total_size_after = 0
        
        # Tính dung lượng trước khi xóa
        for folder_name in folders_to_remove:
            folder_path = Path(folder_name)
            if folder_path.exists():
                for file_path in folder_path.rglob("*"):
                    if file_path.is_file():
                        total_size_before += file_path.stat().st_size
        
        for file_name in files_to_remove:
            file_path = Path(file_name)
            if file_path.exists():
                total_size_before += file_path.stat().st_size
        
        # Xóa thư mục
        removed_folders = []
        for folder_name in folders_to_remove:
            folder_path = Path(folder_name)
            if folder_path.exists():
                try:
                    shutil.rmtree(folder_path)
                    removed_folders.append(folder_name)
                    print(f"Deleted folder: {folder_name}/")
                except Exception as e:
                    print(f"Error deleting folder {folder_name}: {e}")
        
        # Xóa file
        removed_files = []
        for file_name in files_to_remove:
            file_path = Path(file_name)
            if file_path.exists():
                try:
                    file_path.unlink()
                    removed_files.append(file_name)
                    print(f"Deleted file: {file_name}")
                except Exception as e:
                    print(f"Error deleting file {file_name}: {e}")
        
        # Tính dung lượng dataset còn lại
        dataset_folder = Path(self.config.get_dataset_folder())
        if dataset_folder.exists():
            for file_path in dataset_folder.rglob("*"):
                if file_path.is_file():
                    total_size_after += file_path.stat().st_size
        
        # Thống kê
        size_saved = total_size_before - total_size_after
        
        print(f"\nCLEANUP STATISTICS:")
        print(f"Folders deleted: {len(removed_folders)} ({', '.join(removed_folders)})")
        print(f"Files deleted: {len(removed_files)} ({', '.join(removed_files)})")
        print(f"Space saved: {size_saved / (1024*1024):.1f} MB")
        print(f"Dataset size remaining: {total_size_after / (1024*1024):.1f} MB")
        print(f"Only keeping 'dataset/' folder with .wav and .json files")
        
        return True
        
    def show_final_results(self):
        """Hiển thị kết quả cuối cùng"""
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        # Thông tin channel
        if 'crawl' in self.results:
            crawl_info = self.results['crawl']
            print(f"Channel: {crawl_info['channel_name']}")
            print(f"Videos crawled: {crawl_info['total_videos']}")
            print(f"Total duration: {crawl_info['total_duration']/3600:.1f} hours")
        
        # Thống kê pipeline
        print(f"\nPipeline statistics:")
        print(f"- Subtitles downloaded: {self.results.get('subtitles', 0)} files")
        print(f"- Audio segments: {self.results.get('audio_segments', 0)} files")
        
        if 'dataset' in self.results:
            dataset_info = self.results['dataset']
            print(f"- Dataset audio: {dataset_info['audio_files']} files")
            print(f"- Dataset JSON: {dataset_info['json_files']} files")
        
        # Hiển thị mẫu dataset
        dataset_folder = Path(self.config.get_dataset_folder())
        if dataset_folder.exists():
            audio_pattern = f"*.{self.config.get_audio_format()}"
            audio_files = list(dataset_folder.glob(audio_pattern))
            json_files = list(dataset_folder.glob("*.json"))

            print(f"\nDataset folder: {dataset_folder.absolute()}")

            if audio_files:
                print(f"\nSample audio files:")
                for i, audio_file in enumerate(audio_files[:3]):
                    print(f"  {i+1}. {audio_file.name}")
                if len(audio_files) > 3:
                    print(f"  ... and {len(audio_files) - 3} more files")
            
            # Hiển thị mẫu JSON
            if json_files:
                print(f"\nSample JSON content:")
                import json
                try:
                    with open(json_files[0], 'r', encoding='utf-8') as f:
                        sample_json = json.load(f)
                    print(f"File: {json_files[0].name}")
                    print(json.dumps(sample_json, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"Could not read JSON file: {e}")
                    
    def run_pipeline(self):
        """Run complete pipeline with error handling and checkpointing"""
        logger.info("="*60)
        logger.info("YOUTUBE CHANNEL TO SPEECH DATASET PIPELINE")
        logger.info("="*60)
        logger.info(f"Channel URL: {self.channel_url}")
        logger.info(f"Max videos: {self.max_videos}")
        logger.info(f"Checkpoint enabled: {self.enable_checkpoint}")
        logger.info("="*60)

        start_time = time.time()
        pipeline_success = False

        try:
            # Show checkpoint status if resuming
            if self.checkpoint and self.checkpoint.completed_steps:
                logger.info("\nResuming from checkpoint:")
                self.checkpoint.print_status()

            # Step 1: Crawl channel
            if not self.step1_crawl_channel():
                logger.error("ERROR: Could not crawl channel")
                return False

            # Step 2: Download subtitles
            if not self.step2_download_subtitles():
                logger.error("ERROR: Could not download subtitles")
                return False

            # Step 3: Process audio
            if not self.step3_process_audio():
                logger.error("ERROR: Could not process audio")
                return False

            # Step 4: Create labels
            if not self.step4_create_labels():
                logger.error("ERROR: Could not create labels")
                return False

            # Step 5: Validate dataset (non-critical)
            self.step5_validate_dataset()

            # Step 6: Cleanup (if enabled)
            if not self.step6_cleanup_intermediate_files():
                logger.warning("WARNING: Cleanup had issues but continuing")

            # Display results
            self.show_final_results()

            end_time = time.time()
            duration = end_time - start_time

            logger.info(f"\n{'='*60}")
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"{'='*60}")
            logger.info(f"Processing time: {duration/60:.1f} minutes")
            logger.info(f"Dataset ready for model training!")

            # Clear checkpoint on success
            if self.checkpoint:
                self.checkpoint.clear()
                logger.info("Checkpoint cleared after successful completion")

            pipeline_success = True
            return True

        except Exception as e:
            logger.error(f"PIPELINE ERROR: {e}", exc_info=True)
            logger.error("Pipeline failed. Checkpoint saved for resume.")
            return False

        finally:
            # Print final checkpoint status if failed
            if not pipeline_success and self.checkpoint:
                logger.info("\nPipeline incomplete. To resume, run the same command again.")
                self.checkpoint.print_status()

def main():
    """Main function - processes all channels from config.json"""
    print("YouTube Channel to Dataset Pipeline")
    print("="*50)
    print("Fully automated pipeline - all settings from config.json")
    print()

    # Load configuration from JSON
    config = ConfigManager()
    config.print_config_summary()

    # Get channel list from config
    channels = config.get_channels()

    if not channels:
        print("\nERROR: No channels found in config.json!")
        print("Please add channel URLs to the 'channels' array in config.json")
        return
    
    # Thống kê tổng quá trình
    total_success = 0
    total_failed = 0
    failed_channels = []
    
    print(f"\nStarting processing...")
    print("="*60)
    
    for i, channel_url in enumerate(channels, 1):
        print(f"\n{'='*60}")
        print(f"PROCESSING CHANNEL {i}/{len(channels)}")
        print(f"URL: {channel_url}")
        print(f"{'='*60}")
        
        try:
            # Tạo thư mục riêng cho mỗi channel nếu được cấu hình
            original_cwd = os.getcwd()

            if config.should_create_channel_folders():
                channel_name = channel_url.split('@')[-1].split('/')[0] if '@' in channel_url else f"channel_{i}"
                channel_folder = Path(config.get_output_dir()) / channel_name
                channel_folder.mkdir(parents=True, exist_ok=True)

                # Get absolute path BEFORE changing directory
                abs_channel_folder = channel_folder.absolute()

                # Chuyển vào thư mục channel
                os.chdir(channel_folder)
                print(f"Processing in directory: {abs_channel_folder}")

            # Chạy pipeline cho channel này với config
            pipeline = ChannelToDatasetPipeline(channel_url, config)
            success = pipeline.run_pipeline()
            
            # Quay lại thư mục gốc
            os.chdir(original_cwd)
            
            if success:
                total_success += 1
                print(f"\nSUCCESS: Channel {i} completed")
            else:
                total_failed += 1
                failed_channels.append((i, channel_url))
                print(f"\nFAILED: Channel {i} encountered error")
                
        except Exception as e:
            total_failed += 1
            failed_channels.append((i, channel_url))
            print(f"\nERROR: Channel {i} - {e}")
            
            # Đảm bảo quay lại thư mục gốc nếu có lỗi
            try:
                os.chdir(original_cwd)
            except:
                pass
        
        # Thông báo tiến độ
        print(f"\nProgress: {i}/{len(channels)} channels processed")
        print(f"Success: {total_success}, Failed: {total_failed}")

        # Nghỉ giữa các channel để tránh quá tải YouTube
        if i < len(channels):
            delay_seconds = 60  # Increased from 10 to 60 seconds
            print(f"Waiting {delay_seconds} seconds before processing next channel to avoid YouTube rate limiting...")
            time.sleep(delay_seconds)
    
    # Báo cáo cuối cùng
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}")
    print(f"Total channels: {len(channels)}")
    print(f"Success: {total_success}")
    print(f"Failed: {total_failed}")
    
    if failed_channels:
        print(f"\nFailed channels:")
        for idx, url in failed_channels:
            print(f"  {idx}. {url}")
    
    if total_success > 0:
        print(f"\nSuccessfully created datasets saved in 'datasets/' directory")
        print(f"Each channel has its own folder containing .wav and .json files")
    
    print(f"\nAll completed!")

if __name__ == "__main__":
    main()
