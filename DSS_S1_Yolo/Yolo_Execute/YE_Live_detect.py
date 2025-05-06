import cv2
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from DSS_S1_Yolo.Yolo_Execute import YE_get_path
from ultralytics import YOLO
import torch

def main():
    model_path, image_path, image_seg_path, txt_path, mask_path ,point_on_segment_path,new_txt_path,process_image_path,new_txt_path_gaussian= YE_get_path.get_paths()
    
    # Load model
    model = YOLO(model_path)

    # Thiết lập GPU nếu có
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"Đang chạy trên: {device.upper()}")

    # Mở webcam (0 là webcam mặc định)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không đọc được khung hình.")
            break

        # Resize ảnh về kích thước gốc nếu webcam hỗ trợ
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1200)

        # Dự đoán segmentation
        results = model.predict(source=frame, device=device)

        # Vẽ các đường viền segment
        for result in results:
            if result.masks is not None:
                for mask in result.masks.xy:
                    mask = np.array(mask, dtype=np.int32)
                    cv2.polylines(frame, [mask], isClosed=True, color=(0, 255, 0), thickness=2)

        # Hiển thị
        cv2.imshow("YOLOv11 Segmentation - Live", frame)

        # Bấm 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()





