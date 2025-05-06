import open3d as o3d

# Đọc point cloud
ply_file_path = r'D:\Python\DetectSoleShoe\DSS_S3_Mapping\e\3D.ply'
pcd = o3d.io.read_point_cloud(ply_file_path)

# Kiểm tra màu RGB
print("✅ Đã đọc point cloud!")
print(f"Số lượng điểm: {len(pcd.points)}")
print(f"Có màu RGB không? {'Có' if pcd.has_colors() else 'Không'}")

# Hiển thị point cloud với màu
o3d.visualization.draw_geometries(
    [pcd],
    window_name="3D View with Color",
    width=800,
    height=600
)
