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
   
# Load GPU ; Image ; Model Yolo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Đang chạy trên: {device.upper()}")
    model = YOLO(str(model_path)).to(device)
    image = cv2.imread(str(image_path))
    if image is None:
        print("Không thể đọc ảnh!")
        return
    
# RESULT 
    results = model.predict(source=image, imgsz=960, save=False, show=False, device=device)

# SAVE POINT
    with open(txt_path, "w") as f:
        for result_idx, result in enumerate(results):
            if result.masks is not None:
                for obj_idx, mask in enumerate(result.masks.xy):
                    # f.write(f"# Object {obj_idx + 1} (Result {result_idx + 1})\n")
                    for point in mask:
                        x, y = point
                        f.write(f"{int(x)},{int(y)}\n")
                    f.write("\n")
    print(f"Đã lưu các điểm segment vào {txt_path}")

# SHOW RESULT IMAGE
    for result in results:
        if result.masks is not None:
            for mask in result.masks.xy:
                mask = np.array(mask, dtype=np.int32)
                cv2.polylines(image, [mask], isClosed=True, color=(0, 0, 255), thickness=1)

    cv2.imwrite(str(image_seg_path), image)
    cv2.imshow("Segmentation Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()