from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry
import os
import cv2
import numpy as np


def resize_with_padding(image, target_size=(1200, 1200)):
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded, scale, left, top


def resize_to_fit_screen(image, max_size=1000):
    """
    輸出圖片視窗大小自動適應，避免超出螢幕。
    """
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image


def foreground_segmentation_opencv(img_path, mask_generator, display_size=(1200, 1200)):
    image = cv2.imread(img_path)
    if image is None:
        print(f"無法載入圖片: {img_path}")
        return None, None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    padded_img, scale, pad_left, pad_top = resize_with_padding(image, target_size=display_size)
    clone = padded_img.copy()
    selected_points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            orig_x = (x - pad_left) / scale
            orig_y = (y - pad_top) / scale
            if 0 <= orig_x < image.shape[1] and 0 <= orig_y < image.shape[0]:
                selected_points.append([orig_x, orig_y])
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow("Select Points - ESC to Finish", clone)

    cv2.namedWindow("Select Points - ESC to Finish", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Points - ESC to Finish", 1000, 1000)  # 手動設定視窗尺寸
    cv2.imshow("Select Points - ESC to Finish", clone)
    cv2.setMouseCallback("Select Points - ESC to Finish", click_event)

    while True:
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 結束
            break

    cv2.destroyAllWindows()

    if not selected_points:
        print("沒有選取任何點，請重新選擇！")
        return None, None

    input_points = np.array(selected_points)
    input_labels = np.ones(len(input_points), dtype=int)

    mask_generator.set_image(image_rgb)
    masks, scores, logits = mask_generator.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False
    )

    final_mask = masks[0]
    mask_to_save = image.copy()
    mask_to_save[final_mask == 1] = (255, 255, 255)  # 遮罩區塊標記白色
    mask_to_save[final_mask == 0] = (0, 0, 0)  # 遮罩區塊標記黑色

    overlay = image.copy()
    overlay[final_mask == 1] = (0, 255, 0)  # 遮罩區塊標記綠色
    for point in input_points:
        cv2.circle(overlay, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    display_result = resize_to_fit_screen(overlay, max_size=1000)
    cv2.imshow("Segmentation Result", display_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Contour enhancement
    # 原本已經取得 final_mask
    final_mask = final_mask.astype(np.uint8) * 255  # 確保是uint8，cv2需要

    # 用 findContours 擷取輪廓
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 創建全黑背景，背景為黑色
    contour_layout = np.zeros_like(final_mask)

    # 在黑色背景上填充輪廓，讓輪廓內部是白色
    cv2.drawContours(contour_layout, contours, -1, (255), thickness=cv2.FILLED)

    # 將原圖顯示與輪廓的結合，標註提示點
    layout_visual = image.copy()
    for point in input_points:
        cv2.circle(layout_visual, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

    # 縮放並補邊
    contour_layout_pad, _, _, _ = resize_with_padding(contour_layout, target_size=display_size)
    display_result = resize_to_fit_screen(contour_layout_pad, max_size=1000)

    # 顯示結果
    cv2.imshow("Contour Layout (White Inside, Black Outside)", display_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 儲存分割結果
    result_img_path = img_path.replace(base_path, result_path)
    folder = os.path.dirname(result_img_path)
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(result_img_path, contour_layout)
    print(f"分割結果已儲存至: {result_img_path}")

    return final_mask, input_points


def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    dataset_name = "v5_sub_clip"
    # image_folder_name = "20250505_instree_1_image"
    image_folder_name = dataset_name
    base_path = f"/home/jack/Code/Research/instree_analysis/experiment_image/{dataset_name}/{image_folder_name}"
    result_path = f"/home/jack/Code/Research/instree_analysis/experiment_image/{dataset_name}/{image_folder_name}_mask"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    sam = sam_model_registry["vit_b"](checkpoint="/home/jack/Code/Research/segment-anything/sam_vit_b_01ec64.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    for cpt in os.listdir(base_path):
        cpt_path = os.path.join(base_path, cpt, "v0")
        if not os.path.isdir(cpt_path):
            continue

        for img_name in os.listdir(cpt_path):
            img_path = os.path.join(cpt_path, img_name)
            print(f"Processing: {img_path}")
            result_img_path = img_path.replace(base_path, result_path)
            if os.path.exists(result_img_path):
                print(f"已存在結果圖片: {result_img_path}，跳過處理")
                continue
            try:
                mask, points = foreground_segmentation_opencv(img_path, predictor)
                if mask is not None:
                    print(f"成功分割: {img_name}，選取 {len(points)} 個點")
                else:
                    print(f"跳過: {img_name}")
            except Exception as e:
                print(f"錯誤處理圖片 {img_name}: {e}")
