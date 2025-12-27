#!/usr/bin/env python3
"""
Checkpoint Manager
Provides checkpoint/resume functionality for long-running pipeline operations
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages pipeline checkpoints for resume capability"""

    def __init__(self, checkpoint_file: str = "pipeline_checkpoint.json"):
        self.checkpoint_file = Path(checkpoint_file)
        self.checkpoint_data = self.load_checkpoint()

    def load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint if available"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    checkpoint = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return checkpoint
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                return {}
        return {}

    def save_checkpoint(self, data: Dict[str, Any]):
        """Save checkpoint to disk"""
        try:
            checkpoint = {
                **data,
                'last_updated': datetime.now().isoformat(),
                'timestamp': time.time()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def update_checkpoint(self, **kwargs):
        """Update checkpoint with new data"""
        self.checkpoint_data.update(kwargs)
        self.save_checkpoint(self.checkpoint_data)

    def get_checkpoint(self) -> Dict[str, Any]:
        """Get current checkpoint data"""
        return self.checkpoint_data

    def clear_checkpoint(self):
        """Clear checkpoint file"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
                logger.info(f"Cleared checkpoint file {self.checkpoint_file}")
            self.checkpoint_data = {}
        except Exception as e:
            logger.error(f"Error clearing checkpoint: {e}")

    def has_checkpoint(self) -> bool:
        """Check if checkpoint exists"""
        return bool(self.checkpoint_data)


class PipelineCheckpoint:
    """Manages step-by-step pipeline checkpoints"""

    def __init__(self, pipeline_name: str = "default"):
        self.pipeline_name = pipeline_name
        self.checkpoint_file = Path(f"{pipeline_name}_checkpoint.json")
        self.completed_steps = set()
        self.processed_items = set()
        self.failed_items = []
        self.step_results = {}
        self.load()

    def load(self):
        """Load checkpoint from file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.completed_steps = set(data.get('completed_steps', []))
                    self.processed_items = set(data.get('processed_items', []))
                    self.failed_items = data.get('failed_items', [])
                    self.step_results = data.get('step_results', {})
                logger.info(
                    f"Loaded checkpoint: {len(self.completed_steps)} steps completed, "
                    f"{len(self.processed_items)} items processed"
                )
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")

    def save(self):
        """Save checkpoint to file"""
        try:
            # Convert Path objects to strings for JSON serialization
            def convert_paths(obj):
                if isinstance(obj, Path):
                    return str(obj)
                elif isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                return obj

            data = {
                'pipeline_name': self.pipeline_name,
                'completed_steps': list(self.completed_steps),
                'processed_items': list(self.processed_items),
                'failed_items': self.failed_items,
                'step_results': convert_paths(self.step_results),
                'last_updated': datetime.now().isoformat()
            }
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")

    def mark_step_complete(self, step_name: str, result: Any = None):
        """Mark a pipeline step as complete"""
        self.completed_steps.add(step_name)
        if result is not None:
            self.step_results[step_name] = result
        self.save()
        logger.info(f"Step completed: {step_name}")

    def is_step_complete(self, step_name: str) -> bool:
        """Check if a step is already complete"""
        return step_name in self.completed_steps

    def mark_item_processed(self, item_id: str):
        """Mark an item as processed"""
        self.processed_items.add(item_id)
        self.save()

    def is_item_processed(self, item_id: str) -> bool:
        """Check if an item is already processed"""
        return item_id in self.processed_items

    def mark_item_failed(self, item_id: str, error: str):
        """Mark an item as failed"""
        self.failed_items.append({
            'item_id': item_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })
        self.save()

    def get_failed_items(self) -> List[Dict[str, str]]:
        """Get list of failed items"""
        return self.failed_items

    def get_unprocessed_items(self, all_items: List[str]) -> List[str]:
        """Get list of items that haven't been processed yet"""
        return [item for item in all_items if item not in self.processed_items]

    def clear(self):
        """Clear checkpoint"""
        try:
            if self.checkpoint_file.exists():
                self.checkpoint_file.unlink()
            self.completed_steps = set()
            self.processed_items = set()
            self.failed_items = []
            self.step_results = {}
            logger.info(f"Cleared checkpoint for {self.pipeline_name}")
        except Exception as e:
            logger.error(f"Error clearing checkpoint: {e}")

    def print_status(self):
        """Print checkpoint status"""
        print(f"\nCheckpoint Status for {self.pipeline_name}:")
        print(f"  Completed steps: {len(self.completed_steps)}")
        if self.completed_steps:
            print(f"    - {', '.join(sorted(self.completed_steps))}")
        print(f"  Processed items: {len(self.processed_items)}")
        print(f"  Failed items: {len(self.failed_items)}")
        if self.failed_items:
            print(f"    Recent failures:")
            for item in self.failed_items[-5:]:  # Show last 5 failures
                print(f"      - {item['item_id']}: {item['error']}")


def with_checkpoint(checkpoint: PipelineCheckpoint, step_name: str):
    """Decorator to add checkpoint functionality to a function"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if checkpoint.is_step_complete(step_name):
                logger.info(f"Step {step_name} already completed, skipping...")
                return checkpoint.step_results.get(step_name)

            logger.info(f"Executing step: {step_name}")
            result = func(*args, **kwargs)
            checkpoint.mark_step_complete(step_name, result)
            return result
        return wrapper
    return decorator
