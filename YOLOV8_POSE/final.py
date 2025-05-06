from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt

# Load mô hình YOLOv8 Pose đã train xong
model = YOLO("best.pt")

# Đọc ảnh test
image_path = r"D:\Python\Yolov8_pose\dataset\dataset\100.jpg"  # Đổi thành đường dẫn ảnh của bạn
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Dự đoán keypoints
results = model(image)

# Danh sách tên điểm keypoint
kpt_names = ["Head", "Mid", "Heel"]
connections = [(0, 1), (1, 2)]  # Kết nối từ Head → Mid → Heel

for r in results:
    boxes = r.boxes  # Lấy bounding box
    keypoints = r.keypoints  # Lấy các keypoints

    # Vẽ Bounding Box
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ khung xanh

    # Vẽ keypoints & đường nối
    for kpts in keypoints:
        points = kpts.xy[0].cpu().numpy()  # Chuyển keypoints thành numpy array

        # Vẽ điểm keypoints
        for i, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            cv2.circle(image, (x, y), 6, (255, 0, 0), -1)  # Chấm tròn xanh
            cv2.putText(image, kpt_names[i], (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)  # Ghi tên điểm

        # Vẽ đường kết nối giữa các keypoints
        for (i, j) in connections:
            x1, y1 = map(int, points[i])
            x2, y2 = map(int, points[j])
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Đường màu đỏ

# Hiển thị ảnh với matplotlib
plt.imshow(image)
plt.axis("off")
plt.show()
