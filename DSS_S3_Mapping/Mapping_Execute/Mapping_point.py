import numpy as np
import cv2
import open3d as o3d
import re

# --- Cấu hình ---
depth_path = r'D:\Python\DetectSoleShoe\DSS_S3_Mapping\e\imgTexture1.bmp'      # Ảnh depth
segment_txt = r'D:\Python\DetectSoleShoe\DSS_S2_Process\Process_Result\S2_Process_Result_gaussian_point\imgTexture1NEW.txt'  # File chứa các điểm YOLO segment
image = r"D:\Python\DetectSoleShoe\DSS_S3_Mapping\e\3D.ply"
output_ply = r'D:\Python\DetectSoleShoe\DSS_S3_Mapping\Mapping_Result.ply'       # File xuất 3D
depth_scale = 1.0                   # Nếu ảnh depth theo mm thì chia /1000

# Ma trận nội tại (intrinsicMat1 trong file .yaml)
K = np.array(
 [  [2992.8456933,     0,          952.12107631],
    [   0,         2984.68047628 , 648.74729372],
    [   0,            0,          1,       ]
    ])

K_inv = np.linalg.inv(K)

# --- Bước 1: Đọc ảnh depth ---
depth_img = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

# Nếu là ảnh màu thì chỉ lấy 1 kênh
if len(depth_img.shape) == 3:
    depth_img = depth_img[:, :, 0]

# Kiểm tra và xử lý ảnh depth để loại bỏ giá trị không hợp lệ (chẳng hạn như 0 hoặc các giá trị ngoại lệ)
depth_img = np.where(depth_img == 0, np.nan, depth_img)

# --- Bước 2: Đọc các điểm biên dạng YOLO ---
points_uv = []
with open(segment_txt, 'r') as f:
    for line in f:
        tokens = re.split(r'[,\s]+', line.strip())
        if len(tokens) < 2:
            continue
        u, v = map(float, tokens[:2])
        points_uv.append((int(u), int(v)))

# --- Bước 3: Ánh xạ các điểm (u, v) + depth -> 3D ---
points_3d = []
for u, v in points_uv:
    if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
        z = depth_img[v, u] * depth_scale
        if np.isnan(z) or z == 0:
            continue  # Bỏ qua điểm không có độ sâu hợp lệ
        pixel = np.array([u, v, 1])
        xyz = z * (K_inv @ pixel)  # Ánh xạ từ 2D (u, v) sang 3D
        points_3d.append(xyz)

# --- Bước 4: Xuất point cloud 3D ra file .ply ---
points_3d = np.array(points_3d)

# Kiểm tra nếu không có điểm nào hợp lệ
if points_3d.shape[0] == 0:
    print("Không có điểm hợp lệ trong point cloud!")
else:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    o3d.io.write_point_cloud(output_ply, pcd)
    o3d.visualization.draw_geometries([pcd])
    print(f"✅ Hoàn tất! Tổng cộng {len(points_3d)} điểm 3D đã được lưu vào '{output_ply}'")

# --- Đọc và hiển thị lại point cloud từ file .ply ---
ply_file_path = r'D:\Python\DetectSoleShoe\DSS_S3_Mapping\Mapping_Result.ply'
pcd = o3d.io.read_point_cloud(ply_file_path)

# In thông tin cơ bản
print("✅ Đã đọc point cloud!")
print(f"Số lượng điểm: {len(pcd.points)}")

# Hiển thị point cloud với thiết lập kích thước cửa sổ
o3d.visualization.draw_geometries(
    [pcd],
    window_name="Point Cloud",
    width=800,
    height=600
)
