# CausClass
---

## Installation

First, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview)

Then, create a new env

```bash
conda create -n causal_env python=3.10 -y
conda activate causal_env
pip install -r requirements.txt
```

## Infering

Infer pipeline: YOLOv8-MHSA -> CauScientist(AERCA <-> LLM)

### Run YOLOv8-MHSA to get JSON dynamics

```bash
python scripts.extract_dynamics.py [video_path] \
    --model [yolov8 path] \
    --patience [second]
```

### Run CauScientist

TODO: to be updated

## Training

To make the system work in specific environment, it is important to fine tune
the YOLOv8-MHSA model. Steps for finetuning are provided below.

### Data collecting

Data pipeline: YOLO World clipping (detect human) -> YOLOv8-MHSA classifying ->
training data

```bash
python scripts.auto_labeler.py \
    --video [video_path] \
    --output [output_dir, default=data/custom_dataset] \
    --fps [float, default=2.0] \
    --detector [world model path, default=yolov8s-world.pt] \
    --classifier [class model path, default=models/yolov8_mhsa.pt] \
    --conf [class model conf threshold, default=0.2]
```

### Finetuning

TODO: To be updated
