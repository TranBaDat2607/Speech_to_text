"""
Training Configuration Loader
Loads and validates distillation config from YAML
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PathConfig:
    """Path configuration"""
    preprocessed_dataset: str
    teacher_logits_dir: str
    checkpoints_dir: str
    logs_dir: str
    final_model_dir: str
    
    def validate(self):
        """Ensure output directories exist"""
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.final_model_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Model configuration"""
    student_model_name: str = "small"
    student_pretrained: bool = True
    freeze_encoder: bool = False
    teacher_temperature: float = 3.0


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    epochs: int = 20
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    effective_batch_size: int = 8
    
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_gradient_norm: float = 1.0
    
    warmup_steps: int = 200
    lr_schedule: str = "cosine"
    min_learning_rate: float = 1e-6
    
    mixed_precision: bool = True
    
    save_every_n_steps: int = 500
    eval_every_n_steps: int = 250
    keep_last_n_checkpoints: int = 5
    
    early_stopping_patience: int = 6
    early_stopping_metric: str = "wer"


@dataclass
class DistillationConfig:
    """Distillation-specific config"""
    soft_loss_weight: float = 0.7  # alpha
    hard_loss_weight: float = 0.3  # 1 - alpha
    temperature: float = 3.0
    ignore_index: int = -100
    
    enable_mini_batch: bool = True
    mini_batch_size: int = 1000
    auto_cleanup_logits: bool = True


@dataclass
class DataConfig:
    """Data processing config"""
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    max_audio_length: float = 30.0
    max_text_length: int = 448
    language: str = "vi"
    
    train_split: float = 0.95
    random_seed: int = 42


@dataclass
class HardwareConfig:
    """Hardware settings"""
    device: str = "GPU"
    gpu_id: int = 0
    allow_growth: bool = True
    num_workers: int = 4


@dataclass
class Config:
    """Complete configuration"""
    paths: PathConfig
    model: ModelConfig
    training: TrainingConfig
    distillation: DistillationConfig
    data: DataConfig
    hardware: HardwareConfig
    
    project_name: str = "whisper_distillation"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load config from YAML file"""
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # Parse paths
        paths = PathConfig(
            preprocessed_dataset=config_dict['paths']['preprocessed_dataset'],
            teacher_logits_dir=config_dict['paths']['teacher_logits_dir'],
            checkpoints_dir=config_dict['paths']['checkpoints_dir'],
            logs_dir=config_dict['paths']['logs_dir'],
            final_model_dir=config_dict['paths']['final_model_dir']
        )
        
        # Parse model config
        model = ModelConfig(
            student_model_name=config_dict['student']['model_name'],
            student_pretrained=config_dict['student']['pretrained'],
            freeze_encoder=config_dict['student']['freeze_encoder_initially'],
            teacher_temperature=config_dict['teacher']['temperature']
        )
        
        # Parse training config
        training_dict = config_dict['distillation']
        training = TrainingConfig(
            epochs=training_dict['epochs'],
            batch_size=training_dict['batch_size'],
            gradient_accumulation_steps=training_dict['gradient_accumulation_steps'],
            effective_batch_size=training_dict['effective_batch_size'],
            learning_rate=training_dict['learning_rate'],
            weight_decay=training_dict['weight_decay'],
            max_gradient_norm=training_dict['max_gradient_norm'],
            warmup_steps=training_dict['warmup_steps'],
            lr_schedule=training_dict['lr_schedule'],
            min_learning_rate=training_dict['min_learning_rate'],
            mixed_precision=training_dict['mixed_precision'],
            save_every_n_steps=training_dict['save_every_n_steps'],
            eval_every_n_steps=training_dict['eval_every_n_steps'],
            keep_last_n_checkpoints=training_dict['keep_last_n_checkpoints'],
            early_stopping_patience=training_dict['early_stopping_patience'],
            early_stopping_metric=training_dict['early_stopping_metric']
        )
        
        # Parse distillation config
        distillation = DistillationConfig(
            soft_loss_weight=training_dict['soft_loss_weight'],
            hard_loss_weight=training_dict['hard_loss_weight'],
            temperature=training_dict['temperature'],
            enable_mini_batch=training_dict['enable_mini_batch'],
            mini_batch_size=training_dict['mini_batch_size'],
            auto_cleanup_logits=training_dict['auto_cleanup_logits']
        )
        
        # Parse data config
        data_dict = config_dict['data']
        data = DataConfig(
            sample_rate=data_dict['sample_rate'],
            n_mels=data_dict['n_mels'],
            max_audio_length=data_dict['max_audio_length'],
            max_text_length=data_dict['max_text_length'],
            language=data_dict['language'],
            train_split=data_dict['train_split'],
            random_seed=data_dict['random_seed']
        )
        
        # Parse hardware config
        hw_dict = config_dict['hardware']
        hardware = HardwareConfig(
            device=hw_dict['device'],
            gpu_id=hw_dict['gpu_id'],
            allow_growth=hw_dict['allow_growth'],
            num_workers=hw_dict['num_workers']
        )
        
        project_name = config_dict['project']['name']
        
        config = cls(
            paths=paths,
            model=model,
            training=training,
            distillation=distillation,
            data=data,
            hardware=hardware,
            project_name=project_name
        )
        
        # Validate paths
        config.paths.validate()
        
        return config
    
    def print_summary(self):
        """Print configuration summary"""
        print("\n" + "="*60)
        print("Training Configuration Summary")
        print("="*60)
        
        print(f"\n[Project]")
        print(f"  Name: {self.project_name}")
        
        print(f"\n[Model]")
        print(f"  Student: {self.model.student_model_name}")
        print(f"  Pretrained: {self.model.student_pretrained}")
        print(f"  Freeze encoder: {self.model.freeze_encoder}")
        
        print(f"\n[Training]")
        print(f"  Epochs: {self.training.epochs}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.training.effective_batch_size}")
        print(f"  Learning rate: {self.training.learning_rate}")
        print(f"  Warmup steps: {self.training.warmup_steps}")
        print(f"  Mixed precision: {self.training.mixed_precision}")
        
        print(f"\n[Distillation]")
        print(f"  Soft loss weight (α): {self.distillation.soft_loss_weight}")
        print(f"  Hard loss weight: {self.distillation.hard_loss_weight}")
        print(f"  Temperature: {self.distillation.temperature}")
        print(f"  Mini-batch mode: {self.distillation.enable_mini_batch}")
        
        print(f"\n[Hardware]")
        print(f"  Device: {self.hardware.device}")
        print(f"  GPU ID: {self.hardware.gpu_id}")
        print(f"  Workers: {self.hardware.num_workers}")
        
        print("="*60 + "\n")


def load_config(yaml_path: str) -> Config:
    """
    Convenience function to load configuration
    
    Args:
        yaml_path: Path to YAML config file
        
    Returns:
        Config object
    """
    return Config.from_yaml(yaml_path)
