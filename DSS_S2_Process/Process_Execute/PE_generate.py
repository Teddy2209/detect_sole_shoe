import cv2
import numpy as np
import sys
from pathlib import Path
from scipy.interpolate import make_interp_spline
# Thêm thư mục gốc (soles) vào sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from DSS_S1_Yolo.Yolo_Execute import YE_get_path
from scipy.ndimage import gaussian_filter1d

def PE_gen_read_txt(txt_path):
    """"
    Đọc các điểm từ file txt
    Định dạng file: mỗi dòng là 1 điểm có dạng x,y
    """
    points = []                                                         # khởi tạo
    with open(txt_path, 'r') as f:                                     
        for line in f:                                                  #duyệt qua từng dòng
            line = line.strip()                                         #loại bỏ khoảng trắng đầu và cuối dòng
            if line and not line.startswith('#'):
                try:
                    x, y = map(float, line.split(','))                   # chuyển đổi về số thực ; tạo thành list kiểu  [1.2,1.3]
                    points.append([x, y])                                # Thêm điểm      
                except ValueError:                                        # Bỏ qua nếu lỗi định dạng
                    continue
    return points
def PE_gen_find_extreme_point(points):
    """
    Tìm 8 điểm đặc biệt (2 Xmax, 2 Xmin, 2 Ymax, 2 Ymin)
    Trả về dictionary chứa các nhóm điểm
    """
    if len(points) < 2:
        return {}
    
    points_np = np.array(points)                                   
    
    # Tìm giá trị min/max
    x_min_val = np.min(points_np[:, 0])
    x_max_val = np.max(points_np[:, 0])
    y_min_val = np.min(points_np[:, 1])
    y_max_val = np.max(points_np[:, 1])
    
    # Lấy tất cả điểm có giá trị min/max
    x_min_all = points_np[points_np[:, 0] == x_min_val]
    x_max_all = points_np[points_np[:, 0] == x_max_val]
    y_min_all = points_np[points_np[:, 1] == y_min_val]
    y_max_all = points_np[points_np[:, 1] == y_max_val]
    
    # Chọn 2 điểm cho mỗi loại (nếu có đủ)
    extreme_points = {
        'x_min': x_min_all[:2].tolist(),
        'x_max': x_max_all[:2].tolist(),
        'y_min': y_min_all[:2].tolist(),
        'y_max': y_max_all[:2].tolist()
    }
    
    return extreme_points
def PE_gen_get_prev_idx(points, idx, opposite_point):
    """Xác định index điểm điều khiển hợp lệ (tránh bị trùng điểm đối xứng)."""
    if idx == 0:
        return idx +1
    if not np.allclose(points[idx - 1], opposite_point):
        return idx - 1
    if idx + 1 < len(points):
        return idx + 1
    return None
def PE_process_extreme_point(points, extreme_points, threshold=35):
    """
    Xử lý các điểm đặc biệt:
    - len < 2: loại bỏ
    - len == 2: nếu khoảng cách < threshold thì loại bỏ
                nếu >= threshold thì nội suy đường cong
    """
    new_points = points.copy()

    for category, pts in extreme_points.items():
        print(f"Đang xử lý nhóm: {category}, số điểm: {len(pts)}")
        if (1>0):
            if len(pts) < 2:
                # Loại bỏ điểm lẻ
                pt = np.array(pts[0])
                new_points = [p for p in new_points if not np.allclose(p, pt)]
                print(f"Nhóm {category} chỉ có 1 điểm đã bị loại bỏ.")
                continue

            pt1, pt2 = np.array(pts[0]), np.array(pts[1])
            distance = np.linalg.norm(pt1 - pt2)

            if distance < threshold:
                # Loại bỏ 2 điểm gần nhau
                new_points = [p for p in new_points if not (np.allclose(p, pt1) or np.allclose(p, pt2))]
                print(f"Nhóm {category}: Loại bỏ 2 điểm do khoảng cách {distance:.2f} < {threshold}")
                print("đã xoá 2 điểm kì dị","\n" ,pt1.astype(int),"\n" ,pt2.astype(int))
                continue

            # Tìm index gốc trong danh sách hiện tại
            idx1 = next((i for i, p in enumerate(new_points) if np.allclose(p, pt1)), None)
            idx2 = next((i for i, p in enumerate(new_points) if np.allclose(p, pt2)), None)

            if idx1 is None or idx2 is None:
                print(f"Không tìm thấy index của {category} trong danh sách mới.")
                continue

            # Trường hợp nằm đầu danh sách
            if idx1 == 0 or idx2 == 0:
                insert_index = len(new_points)  # chèn vào cuối
                print(f"Nhóm {category} nằm đầu danh sách chèn đường cong vào cuối.")
            # Trường hợp nằm giữa danh sách
            else:
                insert_index = min(idx1, idx2)  # chèn sau điểm nhỏ hơn       
            
            prev_idx1 = PE_gen_get_prev_idx(new_points, idx1, pt2)
            prev_idx2 = PE_gen_get_prev_idx(new_points, idx2, pt1)

            if prev_idx1 is None or prev_idx2 is None:
                print(f"Không thể tìm điểm điều khiển cho nhóm {category} bỏ qua.")
                continue

            prev1 = new_points[prev_idx1]
            prev2 = new_points[prev_idx2]
            print("2 điểm trước đó:", prev1, prev2)         
            # Xóa 2 điểm cũ
            print("đã xoá 2 điểm kì dị","\n" ,pt1.astype(int),"\n" ,pt2.astype(int))
            new_points = [p for p in new_points if not (np.allclose(p, pt1) or np.allclose(p, pt2))]
            # Tạo đường cong nội suy
            mid_point = ((pt1 + pt2) / 2).tolist()
            control_points = np.array([prev1, mid_point, prev2])
            t = [0, 0.5, 1]
            spline = make_interp_spline(t, control_points, k=2)
            t_new = np.linspace(0, 1, 10)
            curve_points = spline(t_new)
            
            
            insert_point = new_points[insert_index-1] if insert_index < len(new_points) else new_points[-1]
            curve_points_sorted = sorted(curve_points, key=lambda p: np.linalg.norm(p - insert_point))

            # Chèn các điểm nội suy theo thứ tự khoảng cách tăng dần
            for curve_point in curve_points_sorted:
                new_points.insert(insert_index, curve_point.tolist())
                insert_index += 1  # Cập nhật vị trí chèn để giữ thứ tự
                distance = np.linalg.norm(curve_point - insert_point)
                print(f"Chèn điểm: {curve_point.tolist()}, khoảng cách: {distance:.2f}")
            print(f"Nhóm {category}: Thay 2 điểm bằng đường cong (d = {distance:.2f}) tại vị trí {insert_index - len(curve_points)}")
        
    return np.array(new_points, dtype=np.int32)

def main():
    model_path, image_path, image_seg_path, txt_path, mask_path ,point_on_segment_path,new_txt_path,process_image_path,new_txt_path_gaussian = YE_get_path.get_paths()
# Đọc ảnh từ đường dẫn
    # image = cv2.imread(result_path)
    # if image is None:
    #     print("Không thể đọc ảnh từ đường dẫn:", result_path)
    # else:
    #     cv2.imshow("i", image)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    # Đọc ảnh gốc để vẽ kết quả
    image = cv2.imread(image_path)
    original_image = image.copy()
    gausian_1D_image = image.copy()
    gaussian_2D_image = image.copy()
    if image is None:
        print("Không thể đọc ảnh!")
        return
    # Đọc ảnh segment để vẽ kết quả

    image_seg = cv2.imread(image_seg_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return
    # Tạo ảnh mask trắng
    mask = np.ones_like(image) * 255

    # Đọc điểm từ file txt
    points = PE_gen_read_txt(txt_path)
    if not points:
        print("Không có điểm nào được đọc từ file!")
        return
    # Tìm các điểm đặc biệt
    extreme_points = PE_gen_find_extreme_point(points)

    # Vẽ và đánh dấu các điểm đặc biệt
    colors = {
        'x_min': (0, 255, 0),   # Xanh lá
        'x_max': (255, 0, 0),   # Đỏ
        'y_min': (0, 255, 255), # Vàng
        'y_max': (255, 0, 255)  # Tím
    }
    
    # Vẽ tất cả các điểm lên ảnh
    for pt in points:
        cv2.circle(image, tuple(map(int, pt)), 2, (0, 0, 255), -1)
        cv2.circle(mask, tuple(map(int, pt)), 2, (0, 0, 255), -1)
        
    for category, pts in extreme_points.items():
        for i, pt in enumerate(pts[:2]):  # Lấy tối đa 2 điểm mỗi loại
            if len(pt) == 2:
                cv2.circle(image_seg, tuple(map(int, pt)), 2, colors[category], -1)
                cv2.circle(mask, tuple(map(int, pt)), 2, colors[category], -1)
                label = f"{category}_{i+1}"
                cv2.putText(image_seg, label, (int(pt[0])+10, int(pt[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[category], 2)
                cv2.putText(mask, label, (int(pt[0])+10, int(pt[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[category], 2)
    
    # Hiển thị thông tin
    print("Các điểm đặc biệt được tìm thấy:")
    for category, pts in extreme_points.items():
        print(f"{category.upper()}: {len(pts)} điểm")
        for i, pt in enumerate(pts):
            print(f"  Điểm {i+1}: ({int(pt[0])}, {int(pt[1])})")
    
    print("EXTREME POINTS:", extreme_points)
    for category, pts in extreme_points.items():
        if len(pts) >= 2:  # Kiểm tra nếu nhóm có ít nhất 2 điểm
            pt1, pt2 = pts[0], pts[1]  # Lấy 2 điểm đầu tiên
            distance = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
            print(f"Khoảng cách giữa hai điểm trong nhóm {category.upper()} là: {int(distance)}")


    # Xử lý các điểm đặc biệt theo yêu cầu
    new_points = PE_process_extreme_point(points, extreme_points)
    # SAVE NEW POINT
    with open(new_txt_path, "w") as f:
        for point in new_points:
            if point is not None:
                x, y = point
                f.write(f"{int(x)},{int(y)}\n")
    print(f"Đã lưu các điểm vào {new_txt_path}")

    smooth_points = gaussian_filter1d(new_points, sigma=3, axis=0)
    with open(new_txt_path_gaussian, "w") as k:
        for point in smooth_points:
         if point is not None:
             x, y = point
             k.write(f"{int(x)},{int(y)}\n")



    gaussian_contour = np.array(smooth_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(gausian_1D_image, [gaussian_contour], isClosed=True, color=(0, 0, 255), thickness=1)
    cv2.imshow("Gaussian 1D", gausian_1D_image)
    # Lưu và hiển thị kết quả
    # cv2.imwrite(str(point_on_segment_path), image_seg)
    # cv2.imwrite(str(mask_path), mask)
    # cv2.imshow("image", image)
    # cv2.imshow("Extreme Points Result", image_seg)
    # cv2.imshow("Extreme Points Mask", mask)
    contour = np.array(new_points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(original_image, [contour], isClosed= True , color=(0, 0, 255), thickness=1)
    cv2.imshow("image_process",original_image)
    cv2.imwrite(str(process_image_path),original_image)
    print(f"Đã lưu các anh vào {process_image_path}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
