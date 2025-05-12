# Human Detection Based on Lightweight Models, Optimized for Cameras Mounted on Robots

## Overview

This repository provides a **human detection tool based on lightweight models** optimized for performing on **cameras mounted on robots**. 

## Installation

```
python3.10 -m venv ./myenv
source ./myenv/bin/activate
pip install -r ./requirements.txt
```

## Download COCO

```
python download_coco.py --year 2017 --target_dir ./datasets
```

## Perform data augmentation on COCO

```
python augment_coco.py --year 2017 --target_dir ./datasets
```
## Transform JSON to YOLO data format

```
python json2yolo.py --year 2017 --target_dir ./datasets
```
## Train YOLO (e.g. YOLO11n)

```
python train_yolo.py --model YOLO11n --yaml_fln  --output_dir
```
