import argparse
import json
import cv2
from collections import defaultdict
from ultralytics import YOLO
import torch
import torch.nn as nn
import ultralytics.nn.modules as modules

def process_video(video_path, model_path="yolov8-mhsa.pt", patience_sec=2.0):
    print(f"--- Đang khởi động YOLO với ByteTrack ---")
    try:
        model = YOLO(model_path)
    except FileNotFoundError:
        print(f"Cảnh báo: Không tìm thấy {model_path}. Dùng tạm yolov8n.pt để test code nhé, nhớ đổi lại sau!")
        model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không mở được video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration = total_frames / fps
    patience_frames = int(patience_sec * fps)

    # Dictionary lưu trữ trạng thái tracking hiện tại
    # Cấu trúc: {track_id: {'class_id': int, 'class_name': str, 'start_frame': int, 'last_seen_frame': int}}
    active_tracks = {}

    # Danh sách lưu các sự kiện vi mô (JSON 1)
    finished_events = []

    print(f"Video: {video_path} | FPS: {fps:.2f} | Thời lượng: {video_duration:.2f}s")
    print(f"Chạy tracking... (Mất tích quá {patience_sec}s sẽ đóng event)")

    # Chạy YOLO với ByteTrack
    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        save=True,
        save_conf=False,
        stream=True,
        verbose=False,
        iou=0.4,
        conf=0.1,
    )

    frame_idx = 0
    for r in results:
        frame_idx += 1
        current_active_ids = set()

        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy()
            class_ids = r.boxes.cls.int().cpu().numpy()

            for box, track_id, cls_id in zip(boxes, track_ids, class_ids):
                track_id = int(track_id)
                cls_id = int(cls_id)
                cls_name = model.names[cls_id]
                current_active_ids.add(track_id)

                if track_id not in active_tracks:
                    # Bắt đầu một event mới
                    active_tracks[track_id] = {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "start_frame": frame_idx,
                        "last_seen_frame": frame_idx
                    }
                else:
                    # ID đã tồn tại
                    if active_tracks[track_id]["class_id"] == cls_id:
                        # Cập nhật thời gian nhìn thấy lần cuối
                        active_tracks[track_id]["last_seen_frame"] = frame_idx
                    else:
                        # Chuyển đổi hành vi (VD: đang reading chuyển sang hand-raising)
                        # Đóng event cũ, mở event mới
                        old_event = active_tracks[track_id]
                        finished_events.append({
                            "object_id": track_id,
                            "action": old_event["class_name"],
                            "start_time": round(old_event["start_frame"] / fps, 2),
                            "end_time": round(old_event["last_seen_frame"] / fps, 2)
                        })
                        active_tracks[track_id] = {
                            "class_id": cls_id,
                            "class_name": cls_name,
                            "start_frame": frame_idx,
                            "last_seen_frame": frame_idx
                        }

        # Dọn dẹp các track bị mất tích quá lâu (Patience mechanism)
        lost_ids = []
        for tid, data in active_tracks.items():
            if tid not in current_active_ids:
                if (frame_idx - data["last_seen_frame"]) > patience_frames:
                    finished_events.append({
                        "object_id": tid,
                        "action": data["class_name"],
                        "start_time": round(data["start_frame"] / fps, 2),
                        "end_time": round(data["last_seen_frame"] / fps, 2)
                    })
                    lost_ids.append(tid)

        for tid in lost_ids:
            del active_tracks[tid]

        if frame_idx % int(fps * 10) == 0:
            print(f"Đã xử lý: {frame_idx}/{total_frames} frames...")

    # Đóng nốt các event còn đang dang dở khi hết video
    for tid, data in active_tracks.items():
        finished_events.append({
            "object_id": tid,
            "action": data["class_name"],
            "start_time": round(data["start_frame"] / fps, 2),
            "end_time": round(data["last_seen_frame"] / fps, 2)
        })

    cap.release()
    print("Tracking xong! Đang chuyển hóa dữ liệu Vĩ mô (Macro-states)...")

    # Tạo JSON 2: Macro-State Aggregation (1 Hz)
    max_seconds = int(video_duration) + 1
    macro_states = []

    for sec in range(max_seconds):
        # Tạo template đếm cho bin hiện tại
        bin_counts = defaultdict(int)

        # Quét các event vi mô xem event nào rơi vào giây hiện tại [sec, sec+1)
        for event in finished_events:
            if event["start_time"] <= sec < event["end_time"]:
                bin_counts[event["action"]] += 1
            # Xử lý case event diễn ra quá nhanh (< 1s) nhưng nằm gọn trong bin này
            elif sec <= event["start_time"] < sec + 1:
                bin_counts[event["action"]] += 1

        # Format đầu ra
        state_dict = {"bin_id_sec": sec}
        state_dict.update(dict(bin_counts))
        macro_states.append(state_dict)

    # Xuất file
    file_prefix = video_path.split('.')[0]

    micro_file = f"{file_prefix}_micro_events.json"
    with open(micro_file, 'w', encoding='utf-8') as f:
        json.dump(finished_events, f, indent=4)

    macro_file = f"{file_prefix}_macro_states.json"
    with open(macro_file, 'w', encoding='utf-8') as f:
        json.dump(macro_states, f, indent=4)

    print(f"\n--- THÀNH CÔNG ---")
    print(f"1. Dữ liệu Vi mô (Từng hành vi cá nhân) đã lưu tại: {micro_file}")
    print(f"2. Dữ liệu Vĩ mô (Timeseries 1Hz) đã lưu tại: {macro_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trích xuất hành vi lớp học từ Video")
    parser.add_argument("video", help="Đường dẫn đến file video cần phân tích")
    parser.add_argument("--model", default="yolov8-mhsa.pt", help="Đường dẫn file trọng số YOLO")
    parser.add_argument("--patience", type=float, default=2.0, help="Số giây kiên nhẫn trước khi cắt đứt 1 event")
    args = parser.parse_args()

    process_video(args.video, args.model, args.patience)
