#!/usr/bin/env python3
"""
Auto-update config paths for WSL environment
Converts Windows paths to WSL mount paths
"""

import platform
import os
import yaml
from pathlib import Path


def is_wsl():
    """Check if running on WSL"""
    return (
        platform.system() == "Linux" and
        "microsoft" in platform.release().lower()
    )


def get_windows_username():
    """Get Windows username from WSL"""
    if is_wsl():
        # Get from environment or default
        win_user = os.getenv("WIN_USER")
        if not win_user:
            # Try to detect from /mnt/c/Users/
            users_dir = Path("/mnt/c/Users")
            if users_dir.exists():
                # Get first non-system user
                for user_dir in users_dir.iterdir():
                    if user_dir.is_dir() and user_dir.name not in ["Public", "Default", "Default User"]:
                        win_user = user_dir.name
                        break
        return win_user or "Admin"
    return None


def windows_to_wsl_path(win_path: str, win_user: str = None) -> str:
    """
    Convert Windows path to WSL mount path
    
    Examples:
        C:\\Users\\Admin\\Desktop -> /mnt/c/Users/Admin/Desktop
        ./data -> /mnt/c/Users/Admin/Desktop/dat301m/Speech_to_text/distillation/data
    """
    if win_user is None:
        win_user = get_windows_username()
    
    # Handle relative paths
    if win_path.startswith("./") or win_path.startswith("../"):
        # Convert to absolute WSL path
        base = f"/mnt/c/Users/{win_user}/Desktop/dat301m/Speech_to_text/distillation"
        return str(Path(base) / win_path)
    
    # Handle absolute Windows paths
    if ":\\" in win_path or ":/" in win_path:
        # Remove drive letter and convert
        path_without_drive = win_path[3:].replace("\\", "/")
        drive = win_path[0].lower()
        return f"/mnt/{drive}/{path_without_drive}"
    
    # Already WSL path or relative
    return win_path


def update_config_for_wsl(config_path: str = "config/distillation_config.yaml"):
    """Update config file with WSL paths"""
    
    print(f"\n{'='*60}")
    print("Updating Config Paths for WSL")
    print(f"{'='*60}\n")
    
    # Check if running on WSL
    if not is_wsl():
        print("Not running on WSL, no changes needed")
        return
    
    win_user = get_windows_username()
    print(f"Windows user: {win_user}")
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Backup original
    backup_path = config_path + ".windows_backup"
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        print(f"Backed up original config to: {backup_path}")
    
    # Update paths
    print("\nUpdating paths:")
    
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                old_path = value
                new_path = windows_to_wsl_path(value, win_user)
                config['paths'][key] = new_path
                print(f"  {key}:")
                print(f"    Old: {old_path}")
                print(f"    New: {new_path}")
    
    # Save updated config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"\n✓ Config updated: {config_path}")
    print(f"{'='*60}\n")


def show_current_paths():
    """Display current paths configuration"""
    
    print(f"\n{'='*60}")
    print("Current Path Configuration")
    print(f"{'='*60}\n")
    
    print(f"Platform: {platform.system()}")
    print(f"WSL: {is_wsl()}")
    
    if is_wsl():
        print(f"Windows User: {get_windows_username()}")
    
    print(f"\nEnvironment Variables:")
    env_vars = [
        'HF_HOME',
        'TRANSFORMERS_CACHE',
        'KERAS_HOME',
        'PIP_CACHE_DIR',
        'PYTHONUSERBASE'
    ]
    
    for var in env_vars:
        value = os.getenv(var, "Not set")
        print(f"  {var}: {value}")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Update config paths for WSL")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--config', default='config/distillation_config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    if args.show:
        show_current_paths()
    else:
        update_config_for_wsl(args.config)
