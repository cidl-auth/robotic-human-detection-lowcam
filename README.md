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
python transform_coco.py --year 2017 --target_dir ./datasets --split train 
python transform_coco.py --year 2017 --target_dir ./datasets --split val 
```
