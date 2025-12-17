import os
import json
import re
from pathlib import Path
from datetime import datetime, timedelta

class DatasetLabeler:
    def __init__(self, subtitles_folder="subtitles", segments_folder="audio_segments", output_folder="dataset"):
        self.subtitles_folder = Path(subtitles_folder)
        self.segments_folder = Path(segments_folder)
        self.output_folder = Path(output_folder)
        self.create_output_folder()
        
    def create_output_folder(self):
        """Tạo thư mục output"""
        self.output_folder.mkdir(exist_ok=True)
        print(f"Thư mục dataset: {self.output_folder.absolute()}")
        
    def parse_srt_time(self, time_str):
        """Chuyển đổi thời gian SRT thành giây"""
        # Format: 00:00:00,000
        time_str = time_str.replace(',', '.')
        try:
            time_obj = datetime.strptime(time_str, '%H:%M:%S.%f')
            total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
            return total_seconds
        except ValueError:
            return 0
            
    def read_srt_file(self, srt_path):
        """Đọc file SRT và trả về danh sách subtitle entries"""
        subtitles = []
        
        try:
            with open(srt_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tách các subtitle block
            blocks = re.split(r'\n\s*\n', content.strip())
            
            for block in blocks:
                lines = block.strip().split('\n')
                if len(lines) >= 3:
                    # Dòng 1: số thứ tự
                    index = lines[0].strip()
                    
                    # Dòng 2: thời gian
                    time_line = lines[1].strip()
                    if '-->' in time_line:
                        start_time_str, end_time_str = time_line.split(' --> ')
                        start_time = self.parse_srt_time(start_time_str.strip())
                        end_time = self.parse_srt_time(end_time_str.strip())
                        
                        # Dòng 3+: text
                        text = ' '.join(lines[2:]).strip()
                        
                        if text:  # Chỉ lấy subtitle có text
                            subtitles.append({
                                'index': index,
                                'start': start_time,
                                'end': end_time,
                                'text': text
                            })
            
            print(f"Đọc được {len(subtitles)} subtitle entries từ {srt_path.name}")
            return subtitles
            
        except Exception as e:
            print(f"Lỗi khi đọc file SRT {srt_path}: {e}")
            return []
            
    def get_text_for_segment(self, subtitles, segment_start, segment_end):
        """Lấy text tương ứng với segment thời gian"""
        segment_texts = []
        
        for subtitle in subtitles:
            # Kiểm tra overlap giữa subtitle và segment
            overlap_start = max(subtitle['start'], segment_start)
            overlap_end = min(subtitle['end'], segment_end)
            
            if overlap_start < overlap_end:  # Có overlap
                segment_texts.append(subtitle['text'])
        
        return ' '.join(segment_texts).strip()
        
    def create_labels_for_video(self, video_id):
        """Tạo labels cho tất cả segments của một video"""
        # Tìm file SRT
        srt_files = list(self.subtitles_folder.glob(f"{video_id}*.srt"))
        if not srt_files:
            print(f"Không tìm thấy file SRT cho video {video_id}")
            return []
            
        srt_file = srt_files[0]
        print(f"Xử lý video {video_id} với file SRT: {srt_file.name}")
        
        # Đọc subtitles
        subtitles = self.read_srt_file(srt_file)
        if not subtitles:
            return []
        
        # Tìm tất cả segments của video này
        segment_files = list(self.segments_folder.glob(f"{video_id}_segment_*.wav"))
        segment_files.sort()  # Sắp xếp theo thứ tự
        
        print(f"Tìm thấy {len(segment_files)} segments cho video {video_id}")
        
        labels_created = []
        
        for segment_file in segment_files:
            # Trích xuất thông tin segment từ tên file
            # Format: {video_id}_segment_{số}.wav
            segment_name = segment_file.stem
            match = re.search(r'_segment_(\d+)$', segment_name)
            
            if match:
                segment_num = int(match.group(1))
                
                # Tính thời gian segment (mỗi segment 30s)
                segment_start = (segment_num - 1) * 30
                segment_end = segment_num * 30
                
                # Lấy text tương ứng
                segment_text = self.get_text_for_segment(subtitles, segment_start, segment_end)
                
                if segment_text:  # Chỉ tạo label nếu có text
                    # Tạo tên file mới cho dataset
                    dataset_wav_name = f"{video_id}_{segment_num:03d}.wav"
                    dataset_json_name = f"{video_id}_{segment_num:03d}.json"
                    
                    # Copy file wav
                    dataset_wav_path = self.output_folder / dataset_wav_name
                    dataset_json_path = self.output_folder / dataset_json_name
                    
                    # Copy file WAV
                    import shutil
                    shutil.copy2(segment_file, dataset_wav_path)
                    
                    # Create JSON label
                    label_data = {
                        "start": segment_start,
                        "end": segment_end,
                        "duration": segment_end - segment_start,
                        "video_id": video_id,
                        "text": segment_text
                    }
                    
                    with open(dataset_json_path, 'w', encoding='utf-8') as f:
                        json.dump(label_data, f, ensure_ascii=False, indent=2)
                    
                    labels_created.append({
                        'wav_file': dataset_wav_name,
                        'json_file': dataset_json_name,
                        'segment_num': segment_num,
                        'text': segment_text
                    })
                    
                    print(f"Tạo: {dataset_wav_name} + {dataset_json_name}")
                else:
                    print(f"Bỏ qua segment {segment_num}: không có text")
        
        return labels_created
        
    def process_all_videos(self):
        """Xử lý tất cả video có segments"""
        # Tìm tất cả video IDs từ segments
        segment_files = list(self.segments_folder.glob("*_segment_*.wav"))
        video_ids = set()
        
        for segment_file in segment_files:
            # Trích xuất video ID từ tên file
            match = re.match(r'(.+)_segment_\d+\.wav$', segment_file.name)
            if match:
                video_ids.add(match.group(1))
        
        print(f"Tìm thấy {len(video_ids)} video IDs: {list(video_ids)}")
        
        all_results = {}
        total_labels = 0
        
        for video_id in video_ids:
            print(f"\n--- Xử lý video {video_id} ---")
            labels = self.create_labels_for_video(video_id)
            all_results[video_id] = labels
            total_labels += len(labels)
            print(f"Tạo {len(labels)} labels cho video {video_id}")
        
        # Lưu summary
        summary = {
            'total_videos': len(video_ids),
            'total_labels': total_labels,
            'videos': all_results,
            'created_at': datetime.now().isoformat()
        }
        
        summary_file = self.output_folder / "dataset_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\nTONG KET:")
        print(f"Videos xu ly: {len(video_ids)}")
        print(f"Tong so labels tao ra: {total_labels}")
        print(f"Dataset luu tai: {self.output_folder}")
        print(f"Summary luu tai: {summary_file}")
        
        return summary

def main():
    print("Dataset Labeler - Tao labels cho audio segments")
    
    # Kiểm tra thư mục cần thiết
    required_folders = ['subtitles', 'audio_segments']
    for folder in required_folders:
        if not os.path.exists(folder):
            print(f"Khong tim thay thu muc: {folder}")
            return
    
    labeler = DatasetLabeler()
    results = labeler.process_all_videos()
    
    if results and results['total_labels'] > 0:
        print(f"\nHoan thanh! Dataset co {results['total_labels']} cap file .wav + .json")
    else:
        print("Khong tao duoc labels nao")

if __name__ == "__main__":
    main()
