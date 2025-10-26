"""
Script to download and save Whisper pretrained weights locally
"""

import os
import torch
import whisper

def download_and_save_whisper_weights(model_name="tiny", save_dir="pretrained_weights"):
    """
    Download Whisper model and save its weights locally
    
    Args:
        model_name: Whisper model size ("tiny", "base", "small", "medium", "large")
        save_dir: Directory to save the weights
    """
    print(f"Downloading Whisper {model_name} model...")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model = whisper.load_model(model_name)
    
    save_path = os.path.join(save_dir, f"whisper_{model_name}.pt")
    
    print(f"Saving weights to {save_path}...")
    torch.save(model.state_dict(), save_path)
    
    print(f"Successfully saved Whisper {model_name} weights!")
    print(f"File size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")
    
    model_dims = {
        "n_mels": model.dims.n_mels,
        "n_audio_ctx": model.dims.n_audio_ctx,
        "n_audio_state": model.dims.n_audio_state,
        "n_audio_head": model.dims.n_audio_head,
        "n_audio_layer": model.dims.n_audio_layer,
        "n_vocab": model.dims.n_vocab,
        "n_text_ctx": model.dims.n_text_ctx,
        "n_text_state": model.dims.n_text_state,
        "n_text_head": model.dims.n_text_head,
        "n_text_layer": model.dims.n_text_layer,
    }
    
    print(f"\nModel dimensions:")
    for key, value in model_dims.items():
        print(f"  {key}: {value}")
    
    return save_path


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_directory = os.path.join(script_dir, "pretrained_weights")
    
    download_and_save_whisper_weights("tiny", save_directory)
