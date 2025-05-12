import json
from collections import defaultdict
import argparse
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
import yaml


def coco91_to_coco80_class():  # converts 80-index (val2014) to 91-index (paper)
    """Converts COCO 91-class index (paper) to 80-class index (2014 challenge)."""
    return [0, 1, 2,3, 4, 5, 6, 7, 8, 9, 10, None, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, None, 24, 25,
            None, None, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, None, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, None, 60, None, None, 61, None, 62, 63, 64, 65, 66, 67, 68,
            69, 70, 71, 72, None, 73, 74, 75, 76, 77, 78, 79, None,]


def convert_coco_json(json_file, save_dir, txt_fln, img_dir, cls91to80=False):
    coco80 = coco91_to_coco80_class()

    with open(json_file) as f:
        data = json.load(f)

    # Create image dict
    images = {"{:g}".format(x["id"]): x for x in data["images"]}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)
    # Write labels file
    with open(txt_fln, "w") as txt_file:
        for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
            img = images[f"{img_id:g}"]
            h, w, f = img["height"], img["width"], img["file_name"]

            bboxes = []
            segments = []
            for ann in anns:
                if ann["iscrowd"]:
                    continue
                # The COCO box format is [top left x, top left y, width, height]
                box = np.array(ann["bbox"], dtype=np.float64)
                box[:2] += box[2:] / 2  # xy top-left corner to center
                box[[0, 2]] /= w  # normalize x
                box[[1, 3]] /= h  # normalize y
                if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                    continue
                cls = coco80[ann["category_id"] - 1] if cls91to80 else ann["category_id"] - 1  # class
                box = [cls] + box.tolist()
                if box not in bboxes:
                    bboxes.append(box)

            # Write
            with open(Path(os.path.join(save_dir, f)).with_suffix(".txt"), "a") as file:
                for i in range(len(bboxes)):
                    line = (*(bboxes[i]),)  # cls, box or segments
                    file.write(("%g " * len(line)).rstrip() % line + "\n")
            txt_file.write(os.path.join('./', img_dir, img["file_name"]) + '\n')

def min_index(arr1, arr2):
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


if __name__ == "__main__":
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

    args = parser.parse_args()
    save_dir = os.path.join(args.target_dir, "labels")


    txt_fln = os.path.join(args.target_dir, 'train' + args.year + '.txt')
    annotation_file = os.path.join(args.target_dir, "annotations", "person_keypoints_train" + str(args.year) + ".json")
    os.makedirs(os.path.join(save_dir, 'train' + args.year), exist_ok=True)
    convert_coco_json(annotation_file, save_dir=os.path.join(save_dir, 'train' + args.year), txt_fln=txt_fln, img_dir= 'images/' + 'train' + args.year)

    txt_fln = os.path.join(args.target_dir, 'val' + args.year + '.txt')
    annotation_file = os.path.join(args.target_dir, "annotations", "person_keypoints_val" + str(args.year) + ".json")
    os.makedirs(os.path.join(save_dir, 'val' + args.year), exist_ok=True)
    convert_coco_json(annotation_file, save_dir=os.path.join(save_dir, 'val' + args.year), txt_fln=txt_fln, img_dir='images/' + 'val' + args.year,)

    yaml_fln = os.path.join(args.target_dir, 'coco' + args.year + '.yaml')
    data_yaml = {
        'names': {
            0: 'person'
        },
        'test': 'val' + args.year + '.txt',  # 20288 of 40670 images
        'val': 'train' + args.year + '.txt',  # val images (relative to 'path') 5000 images
        'train': 'train' + args.year + '.txt',  # train images (relative to 'path') 118287 images
        'path': args.target_dir,  # dataset root dir
    }

    with open(yaml_fln, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)

    txt_fln = os.path.join(args.target_dir, 'train' + args.year + '_augm.txt')
    annotation_file_augm = os.path.join(args.target_dir, "annotations", "person_keypoints_" + 'train' + str(args.year) + "_augm.json")
    os.makedirs(os.path.join(save_dir, 'train' + args.year + '_augm'), exist_ok=True)
    convert_coco_json(annotation_file_augm, save_dir=os.path.join(save_dir, 'train' + args.year + '_augm') , txt_fln=txt_fln, img_dir='images/' + 'train' + args.year, )

    txt_fln = os.path.join(args.target_dir, 'val' + args.year + '_augm.txt')
    annotation_file_augm = os.path.join(args.target_dir, "annotations", "person_keypoints_" + 'val' + str(args.year) + "_augm.json")
    os.makedirs(os.path.join(save_dir, 'val' + args.year + '_augm'), exist_ok=True)
    convert_coco_json(annotation_file_augm, save_dir=os.path.join(save_dir, 'val' + args.year + '_augm'), txt_fln=txt_fln, img_dir='images/' + 'val' + args.year, )

    yaml_fln = os.path.join(args.target_dir, 'coco' + args.year + '_augm.yaml')
    data_yaml = {
        'path': args.target_dir,  # dataset root dir
        'train': 'train' + args.year + '_augm.txt',  # train images (relative to 'path') 118287 images
        'val': 'train' + args.year + '_augm.txt',  # val images (relative to 'path') 5000 images
        'test': 'val' + args.year + '_augm.txt',  # 20288 of 40670 images
        'names': {
            0: 'person'
        }
    }

    with open(yaml_fln, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)