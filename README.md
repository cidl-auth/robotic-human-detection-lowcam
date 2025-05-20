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
python download_coco.py --year 2017 --target_dir ./datasets/coco
```

## Perform data augmentation on COCO

```
python augment_coco.py --year 2017 --target_dir ./datasets/coco
```
## Transform JSON to YOLO data format

```
python json2yolo.py --year 2017 --target_dir ./datasets/coco
```
## Train YOLO (e.g. YOLO11n)

```
python train_yolo.py --model YOLO11n --yaml_fln ./datasets/coco/coco2017_augm.yaml \
--output_dir ./out/yolo11n
```
## Run inference on a signle image

```
python inference_yolo.py --weights <path_to_yolo11_weights> --source <path_to_img>
```


## Acknowledgments

This work has received partial funding from the Hellenic Foundation for Research & Innovation (H.F.R.I.) scholarship under grant agreement No 20490 (Deep Learning Methodologies for Trustworthy Intelligent Systems) and the research project ”Robotic Safe Adaptation In Unprecedented Situations (RoboSAPIENS)”, which is implemented in the framework of Horizon Europe 2021-2027 research and innovation programme under grant agreement No 101133807. This publication reflects the authors’ views only. The European Commission are not responsible for any use that may be made of the information it contains.

<p align="center">
<img src="https://robosapiens-eu.tech/wp-content/uploads/elementor/thumbs/europeanlogo-r2pgiiytkuoyehehz8416uvf52f3bxv6hkmztxq1am.jpeg"
     alt="EU Flag"
     width="120" />
</p>
<p align="center">
Learn more about <a href="https://robosapiens-eu.tech/">RoboSAPIENS</a>.
</p>
<p align="center">
<img src="https://robosapiens-eu.tech/wp-content/uploads/elementor/thumbs/full_robot2-r2pgiiysrgwxbxr64iuhw1nmi4k46lxqopg2va2qos.png"
     alt="RoboSAPIENS Logo"
     width="80" />     
</p>
