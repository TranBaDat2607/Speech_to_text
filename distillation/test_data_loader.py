import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from data.dataset_loader import load_config, create_dataloaders


def test_data_loader():
    print("\n" + "="*60)
    print("TEST DATA LOADER")
    print("="*60 + "\n")
    
    try:
        print("1. Loading config...")
        config = load_config("config/distillation_config.yaml")
        print(f"   ✓ Config loaded")
        print(f"   - Data dir: {config['paths']['preprocessed_dataset']}")
        print(f"   - Batch size: {config['teacher']['batch_size']}")
        print(f"   - Sample rate: {config['data']['sample_rate']}")
        
        print("\n2. Creating dataloaders...")
        train_loader, val_loader = create_dataloaders(config)
        print(f"   ✓ Dataloaders created")
        
        print("\n3. Dataset info:")
        print(f"   - Train samples: {len(train_loader.dataset)}")
        print(f"   - Val samples: {len(val_loader.dataset)}")
        print(f"   - Train batches: {len(train_loader)}")
        print(f"   - Val batches: {len(val_loader)}")
        
        print("\n4. Testing batch loading...")
        batch = next(iter(train_loader))
        
        print(f"   ✓ Batch loaded successfully")
        print(f"   - Audio shape: {batch['audio'].shape}")
        print(f"   - Batch size: {len(batch['text'])}")
        print(f"   - Duration shape: {batch['duration'].shape}")
        
        print("\n5. Sample data:")
        print(f"   - Text (first): {batch['text'][0][:100]}...")
        print(f"   - Duration (first): {batch['duration'][0]:.2f}s")
        print(f"   - Audio path: {Path(batch['audio_paths'][0]).name}")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60 + "\n")
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n✗ ERROR: {e}")
        print("\nHãy kiểm tra:")
        print("  1. Dataset đã download chưa?")
        print("  2. Path trong config đúng chưa?")
        print("  3. Manifest files tồn tại chưa?")
        return False
        
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_data_loader()
    sys.exit(0 if success else 1)
