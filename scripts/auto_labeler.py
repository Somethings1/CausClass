import os
import cv2
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Nạp "não" MHSA để lừa PyTorch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.mhsa_core import MHSA, patch_yolo_mhsa
setattr(sys.modules[__name__], 'MHSA', MHSA)

def generate_auto_labels(video_path, output_dir, detector_weights, classifier_weights, extract_fps):
    patch_yolo_mhsa()

    # Tạo 3 thư mục: Ảnh gốc, Nhãn text, và Ảnh để nhìn bằng mắt
    images_dir = Path(output_dir) / "images" / "train"
    labels_dir = Path(output_dir) / "labels" / "train"
    visualize_dir = Path(output_dir) / "visualize" / "train"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    visualize_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- KHỞI ĐỘNG HỆ THỐNG AUTO-LABELING (SMART CROP + VISUALIZE) ---")
    detector = YOLO(detector_weights)
    classifier = YOLO(classifier_weights)

    # Lấy từ điển nhãn của MHSA (VD: {0: 'hand-raising', 1: 'reading'...})
    class_names = classifier.names

    cap = cv2.VideoCapture(video_path)
    video_name = Path(video_path).stem
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(orig_fps / extract_fps))

    frame_count = 0
    saved_count = 0
    pbar = tqdm(total=total_frames, desc="Đang soi Video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame_count += 1
        pbar.update(1)

        if frame_count % frame_skip != 0: continue

        h_img, w_img = frame.shape[:2]

        # 1. Tìm người bằng YOLO gốc
        world_res = detector.predict(frame, classes=[0], conf=0.15, verbose=False)
        boxes_world = world_res[0].boxes

        if len(boxes_world) == 0: continue

        temp_labels = []
        # Tạo một bản sao của frame để vẽ vời lên đó, tránh làm bẩn ảnh gốc
        vis_frame = frame.copy()

        # 2. Xử lý từng đứa học sinh (Smart Crop)
        for box in boxes_world:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            bw = x2 - x1
            bh = y2 - y1

            # Bơm phồng vùng cắt thêm 30% để lấy bối cảnh
            pad_x = int(bw * 0.3)
            pad_y = int(bh * 0.3)

            crop_x1 = max(0, x1 - pad_x)
            crop_y1 = max(0, y1 - pad_y)
            crop_x2 = min(w_img, x2 + pad_x)
            crop_y2 = min(h_img, y2 + pad_y)

            crop_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

            if crop_img.shape[0] < 20 or crop_img.shape[1] < 20: continue

            # 3. Ném cho MHSA đoán
            cls_res = classifier.predict(crop_img, conf=0.15, verbose=False)

            if len(cls_res[0].boxes) > 0:
                best_pred = cls_res[0].boxes[0]
                class_id = int(best_pred.cls[0].item())
                action_name = class_names[class_id]

                # Tính tọa độ YOLO để lưu file txt
                x_center = ((x1 + x2) / 2) / w_img
                y_center = ((y1 + y2) / 2) / h_img
                box_width = bw / w_img
                box_height = bh / h_img

                temp_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

                # --- VẼ LÊN ẢNH VISUALIZE ---
                # Vẽ khung chữ nhật màu xanh lá (0, 255, 0)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Viết chữ màu đỏ (0, 0, 255), nền font chữ có thể tinh chỉnh nếu muốn
                cv2.putText(vis_frame, f"{action_name}", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 4. Lưu lại 3 thứ: Ảnh gốc, File txt, và Ảnh Visualize
        if len(temp_labels) > 0:
            img_name = f"{video_name}_f{frame_count:05d}.jpg"
            label_name = f"{video_name}_f{frame_count:05d}.txt"

            cv2.imwrite(str(images_dir / img_name), frame)        # Lưu ảnh sạch
            cv2.imwrite(str(visualize_dir / img_name), vis_frame) # Lưu ảnh đã vẽ hộp
            with open(labels_dir / label_name, 'w') as f_txt:     # Lưu file nhãn
                f_txt.writelines(temp_labels)

            saved_count += 1

    pbar.close()
    cap.release()
    print(f"\n--- THÀNH CÔNG ---")
    print(f"Đã trích xuất {saved_count} khung hình.")
    print(f"👉 Mở thư mục '{output_dir}/visualize/train' ra mà xem thành quả đi!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="data/new_school_dataset")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--detector", default="yolov8m.pt")
    parser.add_argument("--classifier", default="models/yolov8_mhsa.pt")
    args = parser.parse_args()

    generate_auto_labels(args.video, args.output, args.detector, args.classifier, args.fps)
