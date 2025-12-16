#!/usr/bin/env python3
"""
File Validators
Utilities for validating downloaded and processed files
"""

import os
import json
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class FileValidator:
    """Validates file integrity and content"""

    @staticmethod
    def validate_file_exists(file_path: Path, min_size: int = 0) -> Tuple[bool, Optional[str]]:
        """
        Validate that file exists and meets minimum size requirement

        Args:
            file_path: Path to file
            min_size: Minimum file size in bytes

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not file_path.exists():
            return False, f"File does not exist: {file_path}"

        if not file_path.is_file():
            return False, f"Path is not a file: {file_path}"

        file_size = file_path.stat().st_size
        if file_size < min_size:
            return False, f"File too small: {file_size} bytes < {min_size} bytes"

        return True, None

    @staticmethod
    def validate_audio_file(file_path: Path, min_size: int = 1024) -> Tuple[bool, Optional[str]]:
        """
        Validate audio file

        Args:
            file_path: Path to audio file
            min_size: Minimum file size in bytes (default 1KB)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file exists and size
        is_valid, error = FileValidator.validate_file_exists(file_path, min_size)
        if not is_valid:
            return False, error

        # Check extension
        valid_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
        if file_path.suffix.lower() not in valid_extensions:
            return False, f"Invalid audio extension: {file_path.suffix}"

        # Try to load with pydub if available
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(file_path)
            duration_ms = len(audio)

            if duration_ms < 100:  # Less than 0.1 second
                return False, f"Audio too short: {duration_ms}ms"

            logger.debug(f"Audio file validated: {file_path.name} ({duration_ms}ms)")
            return True, None

        except ImportError:
            # If pydub not available, just check file size and extension
            logger.debug("pydub not available, skipping advanced audio validation")
            return True, None

        except Exception as e:
            return False, f"Error reading audio file: {e}"

    @staticmethod
    def validate_srt_file(file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate SRT subtitle file

        Args:
            file_path: Path to SRT file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file exists
        is_valid, error = FileValidator.validate_file_exists(file_path, min_size=10)
        if not is_valid:
            return False, error

        # Check extension
        if file_path.suffix.lower() not in {'.srt', '.vtt'}:
            return False, f"Invalid subtitle extension: {file_path.suffix}"

        # Try to read and parse basic structure
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for basic SRT structure
            if '-->' not in content:
                return False, "Invalid SRT format: missing time markers"

            # Count subtitle entries
            entries = content.strip().split('\n\n')
            if len(entries) < 1:
                return False, "No subtitle entries found"

            logger.debug(f"SRT file validated: {file_path.name} ({len(entries)} entries)")
            return True, None

        except UnicodeDecodeError:
            return False, "File encoding error: not valid UTF-8"

        except Exception as e:
            return False, f"Error reading SRT file: {e}"

    @staticmethod
    def validate_json_file(file_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON file

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check file exists
        is_valid, error = FileValidator.validate_file_exists(file_path, min_size=2)
        if not is_valid:
            return False, error

        # Check extension
        if file_path.suffix.lower() != '.json':
            return False, f"Invalid JSON extension: {file_path.suffix}"

        # Try to parse JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if not isinstance(data, (dict, list)):
                return False, "JSON must be object or array"

            logger.debug(f"JSON file validated: {file_path.name}")
            return True, None

        except json.JSONDecodeError as e:
            return False, f"Invalid JSON format: {e}"

        except Exception as e:
            return False, f"Error reading JSON file: {e}"

    @staticmethod
    def validate_dataset_pair(wav_path: Path, json_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate a dataset pair (WAV + JSON)

        Args:
            wav_path: Path to WAV file
            json_path: Path to JSON label file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate WAV file
        is_valid, error = FileValidator.validate_audio_file(wav_path)
        if not is_valid:
            return False, f"WAV validation failed: {error}"

        # Validate JSON file
        is_valid, error = FileValidator.validate_json_file(json_path)
        if not is_valid:
            return False, f"JSON validation failed: {error}"

        # Check that stems match
        if wav_path.stem != json_path.stem:
            return False, f"File name mismatch: {wav_path.stem} != {json_path.stem}"

        # Validate JSON content
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                label_data = json.load(f)

            # Check required fields
            required_fields = ['text', 'start', 'end']
            missing_fields = [field for field in required_fields if field not in label_data]

            if missing_fields:
                return False, f"Missing required fields in JSON: {missing_fields}"

            # Check text is not empty
            if not label_data['text'].strip():
                return False, "Empty text in JSON label"

            # Check time values are valid
            if label_data['start'] < 0 or label_data['end'] <= label_data['start']:
                return False, f"Invalid time range: {label_data['start']} - {label_data['end']}"

            return True, None

        except Exception as e:
            return False, f"Error validating dataset pair: {e}"


class DatasetValidator:
    """Validates entire dataset folders"""

    def __init__(self, dataset_folder: Path):
        self.dataset_folder = Path(dataset_folder)
        self.validation_results = {
            'total_wav': 0,
            'total_json': 0,
            'valid_pairs': 0,
            'invalid_files': [],
            'missing_pairs': [],
            'orphaned_files': []
        }

    def validate_dataset(self) -> dict:
        """
        Validate entire dataset folder

        Returns:
            Dictionary with validation results
        """
        if not self.dataset_folder.exists():
            logger.error(f"Dataset folder does not exist: {self.dataset_folder}")
            return self.validation_results

        # Get all WAV and JSON files
        wav_files = {f.stem: f for f in self.dataset_folder.glob("*.wav")}
        json_files = {f.stem: f for f in self.dataset_folder.glob("*.json")}

        self.validation_results['total_wav'] = len(wav_files)
        self.validation_results['total_json'] = len(json_files)

        # Find missing pairs
        wav_stems = set(wav_files.keys())
        json_stems = set(json_files.keys())

        missing_json = wav_stems - json_stems
        missing_wav = json_stems - wav_stems

        for stem in missing_json:
            self.validation_results['missing_pairs'].append({
                'file': wav_files[stem].name,
                'type': 'missing_json'
            })
            self.validation_results['orphaned_files'].append(str(wav_files[stem]))

        for stem in missing_wav:
            self.validation_results['missing_pairs'].append({
                'file': json_files[stem].name,
                'type': 'missing_wav'
            })
            self.validation_results['orphaned_files'].append(str(json_files[stem]))

        # Validate pairs
        common_stems = wav_stems & json_stems
        for stem in common_stems:
            wav_path = wav_files[stem]
            json_path = json_files[stem]

            is_valid, error = FileValidator.validate_dataset_pair(wav_path, json_path)

            if is_valid:
                self.validation_results['valid_pairs'] += 1
            else:
                self.validation_results['invalid_files'].append({
                    'wav': str(wav_path),
                    'json': str(json_path),
                    'error': error
                })
                logger.warning(f"Invalid pair {stem}: {error}")

        return self.validation_results

    def print_validation_report(self):
        """Print validation report"""
        results = self.validation_results

        print("\n" + "="*60)
        print("DATASET VALIDATION REPORT")
        print("="*60)
        print(f"Dataset folder: {self.dataset_folder}")
        print(f"\nFile counts:")
        print(f"  WAV files: {results['total_wav']}")
        print(f"  JSON files: {results['total_json']}")
        print(f"  Valid pairs: {results['valid_pairs']}")

        if results['missing_pairs']:
            print(f"\nMissing pairs: {len(results['missing_pairs'])}")
            for item in results['missing_pairs'][:10]:
                print(f"  - {item['file']} ({item['type']})")
            if len(results['missing_pairs']) > 10:
                print(f"  ... and {len(results['missing_pairs']) - 10} more")

        if results['invalid_files']:
            print(f"\nInvalid files: {len(results['invalid_files'])}")
            for item in results['invalid_files'][:10]:
                print(f"  - {Path(item['wav']).name}: {item['error']}")
            if len(results['invalid_files']) > 10:
                print(f"  ... and {len(results['invalid_files']) - 10} more")

        if results['orphaned_files']:
            print(f"\nOrphaned files: {len(results['orphaned_files'])}")

        # Overall status
        total_expected_pairs = results['total_wav']
        success_rate = (results['valid_pairs'] / total_expected_pairs * 100) if total_expected_pairs > 0 else 0
        print(f"\nOverall status: {success_rate:.1f}% valid")

        if success_rate == 100:
            print("Status: PASSED - All files are valid")
        elif success_rate >= 95:
            print("Status: WARNING - Most files are valid")
        else:
            print("Status: FAILED - Many invalid files detected")

        print("="*60)

        return results
