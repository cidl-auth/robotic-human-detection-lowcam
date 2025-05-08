from pycocotools.coco import COCO
import json
import os
import cv2
import numpy as np
import argparse


def augment_coco(image_dir, annotation_file):
    # Initialize COCO API
    coco = COCO(annotation_file)
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)

    # Get category ID for "person" (usually ID=1)
    category_id = 1

    # Get all image IDs that contain persons
    image_ids = coco.getImgIds(catIds=[category_id])

    next_image_id = max(image_ids) + 1
    next_annotation_id = max([ann for ann in coco.anns]) + 1
    # Loop through all images
    for image_id in image_ids:
        # Get image metadata
        image_info = coco.loadImgs(image_id)[0]

        # Get annotations for the current image
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], catIds=[category_id])).copy()

        # Parse annotations
        parsed_annotations = []
        for ann in annotations:
            parsed_annotations.append({
                "image_id": ann["image_id"],
                "file_name": image_info["file_name"],
                "width": image_info["width"],
                "height": image_info["height"],
                "bbox": ann["bbox"],  # [x, y, width, height]
                "keypoints": ann["keypoints"],  # [x1, y1, v1, ..., x17, y17, v17]
                "num_keypoints": ann["num_keypoints"],  # Number of visible keypoints
            })
        image_info = coco.loadImgs(image_id)[0]
        image_path = f"{image_dir}/{image_info['file_name']}"
        image = cv2.imread(image_path)
        mask_a = np.zeros([image.shape[0], image.shape[1], 1])
        mask_b = np.zeros([image.shape[0], image.shape[1], 1])
        create_flag = False
        anns = []
        cnt_id = 0
        for ann in annotations:
            keypoints = np.array(ann["keypoints"]).reshape(-1, 3)  # Reshape to (17, 3)
            bbox = ann["bbox"] # Bounding box (x, y, width, height)
            if (keypoints[11, 2] > 0 or keypoints[12, 2] > 0) and (keypoints[13, 2] > 0 or keypoints[14, 2] > 0):
                x, y, w, h = map(int, bbox)
                n_keypoints = keypoints[11:][keypoints[11:][:, 2] >0, :]
                x_keep = np.min(n_keypoints[:, 0])
                y_keep = np.min(n_keypoints[:, 1])
                w_keep = np.max(n_keypoints[:, 0]) - x_keep
                x_keep = int(max(0, x_keep - 0.2 *w_keep))
                y_keep = int(y_keep)
                x_2_keep = int(min(image_info["width"], x_keep + 1.4 *w_keep))
                w_keep = int(x_2_keep - x_keep)
                h_keep = y + h - y_keep
                h_keep = int(h_keep)
                mask_a[y_keep:y_keep+h_keep, x_keep:x_keep+w_keep] = 1
                mask_b[y:y+h, x:x+w] = 1
                bbox = [x_keep, y_keep, w_keep, h_keep]
                create_flag = True
            else:
                mask_a[int(ann["bbox"][1]):int(ann["bbox"][1]+ann["bbox"][3]), int(ann["bbox"][0]):int(ann["bbox"][0]+ann["bbox"][2])] = 1
            ann['bbox'] = bbox
            ann['image_id'] = next_image_id
            ann['id'] = next_annotation_id + cnt_id
            cnt_id = cnt_id + 1
            anns.append(ann)
        mask_b = mask_b - mask_a
        image[mask_b.squeeze(-1)==1] = 0
        if create_flag:
            img_fln = image_info['file_name'].replace(image_info['file_name'].split('_')[-1], "")
            img_fln = img_fln + str(next_image_id).zfill(12) + '.jpg'
            new_image = {
                "id": next_image_id,
                "file_name": img_fln,
                "height": image_info['height'],
                "width": image_info['width'],
            }
            coco_data['images'].append(new_image)
            cv2.imwrite(f"{image_dir}/{img_fln}", image)
            coco_data['annotations'].extend(anns)
            next_image_id = next_image_id + 1
            next_annotation_id = next_annotation_id + cnt_id

    new_annotation_file = annotation_file.split('.json')[0] + '_augm.json'
    with open(new_annotation_file, "w") as f:
        json.dump(coco_data, f, indent=4)
    print("All COCO images processed successfully!")

def main():
    parser = argparse.ArgumentParser(description="Download and extract COCO dataset.")
    parser.add_argument(
        "--target_dir",
        type=str,
        required=True,
        help="Path to the directory where the dataset will be saved."
    )
    parser.add_argument(
        "--year",
        type=str,
        required=False,
        choices=["2015", "2017"],
        default="2017",
        help="COCO years."
    )
    parser.add_argument(
        "--split",
        type=str,
        required=False,
        choices=["train", "val"],
        default="train",
        help="split to perform augmentation"
    )
    args = parser.parse_args()

    # Path to the COCO keypoint annotation file
    annotation_file = os.path.join(args.target_dir, "annotations", "person_keypoints_" + args.split + str(args.year) + ".json")
    # Path to the COCO dir of images
    image_dir = os.path.join(args.target_dir, args.split + str(args.year))
    augment_coco(image_dir, annotation_file)



if __name__ == "__main__":
    main()