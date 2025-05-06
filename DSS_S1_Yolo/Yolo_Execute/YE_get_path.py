from pathlib import Path

def get_paths():
    base_dir = Path(__file__).parent
    model_path = base_dir.parent /"Yolo_best_file"/ "best_seg_24_4_25.pt"
    image_path = base_dir.parent.parent / "DSS_S3_Mapping" /"e"/ "imgTexture1.bmp"
# tạo path lưu ảnh
    original_image_stem = image_path.stem
    original_image_suffix = image_path.suffix
    image_seg = original_image_stem + "_segment" + original_image_suffix
    image_seg_path = base_dir.parent / "Yolo_Results"/ "S1_Yolo_Result_image" / image_seg

# Tạp path lưu file txt
    filename_txt = image_path.stem + ".txt"
    txt_path = base_dir.parent/ "Yolo_Results"/ "S1_Yolo_Result_point" / filename_txt

# Tạo path lưu mask chưa điểm kì dị
    mask = original_image_stem + "_mask" + original_image_suffix
    mask_path = base_dir.parent.parent/ "DSS_S2_Process"/"Process_Result"/"S2_Process_Result_image" / mask

    point_on_segment = original_image_stem + "_point_in_segment" + original_image_suffix
    point_on_segment_path = base_dir.parent.parent / "DSS_S2_Process"/"Process_Result"/"S2_Process_Result_image"/point_on_segment
# path lưu txt mới
    filename_new_txt = image_path.stem + "NEW.txt"
    new_txt_path =  base_dir.parent.parent/ "DSS_S2_Process"/"Process_Result"/"S2_Process_Result_point" / filename_new_txt
    new_txt_path_gaussian =  base_dir.parent.parent/ "DSS_S2_Process"/"Process_Result"/"S2_Process_Result_gaussian_point" / filename_new_txt
# path new process image
    process_image_stem = image_path.stem 
    process_image_suffix = image_path.suffix
    process_image = process_image_stem + "_processed" + process_image_suffix
    process_image_path =  base_dir.parent.parent/"DSS_S2_Process"/"Process_Result"/"S2_Process_Result_point"/process_image
    return model_path, image_path, image_seg_path, txt_path, mask_path ,point_on_segment_path,new_txt_path,process_image_path,new_txt_path_gaussian
