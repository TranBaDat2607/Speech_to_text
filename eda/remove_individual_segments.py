import json
import os
from pathlib import Path

def remove_individual_segments_from_json(file_path):
    """
    Đọc file JSON và xóa các trường 'individual_segments', 'segment_id', 'duration' nếu có
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Danh sách các trường cần xóa
        fields_to_remove = ['individual_segments', 'segment_id', 'duration']
        removed_fields = []
        
        # Xóa các trường nếu tồn tại
        for field in fields_to_remove:
            if field in data:
                del data[field]
                removed_fields.append(field)
        
        # Nếu có trường nào được xóa thì ghi lại file
        if removed_fields:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"✓ Removed {', '.join(removed_fields)} from: {file_path}")
            return True
        else:
            print(f"- No target fields found in: {file_path}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing file {file_path}: {str(e)}")
        return False

def process_all_json_files(root_directory):
    """
    Duyệt qua tất cả các folder và xử lý các file JSON
    """
    root_path = Path(root_directory)
    total_files = 0
    processed_files = 0
    
    print(f"Starting to process JSON files in: {root_directory}")
    print("-" * 60)
    
    # Duyệt qua tất cả các folder con
    for folder in root_path.iterdir():
        if folder.is_dir() and folder.name != 'eda':  # Bỏ qua folder eda
            print(f"\nProcessing folder: {folder.name}")
            
            # Duyệt qua tất cả file JSON trong folder
            json_files = list(folder.glob("*.json"))
            
            if not json_files:
                print(f"  No JSON files found in {folder.name}")
                continue
                
            for json_file in json_files:
                total_files += 1
                if remove_individual_segments_from_json(json_file):
                    processed_files += 1
    
    print("\n" + "=" * 60)
    print(f"Completed! Processed {processed_files}/{total_files} JSON files")
    print("=" * 60)

if __name__ == "__main__":
    # Đường dẫn đến folder chứa các folder con với file JSON
    root_directory = Path(__file__).parent.parent / "crawl_data" / "output_segments_grouped"
    
    print("Script to remove 'individual_segments', 'segment_id', 'duration' fields from JSON files")
    print("=" * 60)
    
    # Xác nhận trước khi chạy
    response = input(f"Are you sure you want to remove 'individual_segments', 'segment_id', 'duration' fields from all JSON files in {root_directory}? (y/n): ")
    
    if response.lower() in ['y', 'yes', 'có']:
        process_all_json_files(root_directory)
    else:
        print("Operation cancelled.")
