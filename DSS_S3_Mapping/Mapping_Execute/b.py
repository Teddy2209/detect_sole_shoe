import open3d as o3d

# Đọc tệp từ API VS
pcd_vs = o3d.io.read_point_cloud(r"D:\Python\DetectSoleShoe\DSS_S3_Mapping\a\3D.ply")
print(f"Số lượng điểm: {len(pcd_vs.points)}")
print(f"Có màu RGB không? {'Có' if pcd_vs.has_colors() else 'Không'}")

# Đọc tệp từ Accupick
pcd_accupick = o3d.io.read_point_cloud(r"D:\Python\DetectSoleShoe\DSS_S3_Mapping\e\3D.ply")
print(f"Số lượng điểm: {len(pcd_accupick.points)}")
print(f"Có màu RGB không? {'Có' if pcd_accupick.has_colors() else 'Không'}")

# Downsample point cloud từ API VS
pcd_vs_downsampled = pcd_vs.voxel_down_sample(voxel_size=0.01)

# Hiển thị point cloud sau khi giảm số lượng điểm
o3d.visualization.draw_geometries([pcd_vs], window_name="Downsampled API VS Point Cloud")
o3d.visualization.draw_geometries([pcd_accupick], window_name="Downsampled acc VS Point Cloud")