# SCSE-NET:Structure–Context–Scale Collaborative Enhancement Network for Infrared Small Target Detection

This repository provides the implementation of **SCSE-Net**, a Structure–Context–Scale Collaborative Enhancement Network for infrared small target detection.

SCSE-Net is designed for infrared images with extremely small target sizes, weak texture information, complex background interference, and severe feature attenuation during feature propagation. The method enhances small target representation from three complementary perspectives by introducing an **Orientation-Aware Shallow Structure Module (OALS)**, a **Context Routing Gated Module (CRGM)**, and an **Adaptive Granularity Multi-Scale Module (AGMS)**. OALS preserves shallow edge, contour, and directional structural information before early downsampling, CRGM improves target–background discrimination through multi-granularity contextual modeling and dynamic gating, and AGMS strengthens cross-scale feature adaptation through multi-receptive-field feature recalibration.

Compared with the DEIM-S baseline, SCSE-Net achieves better detection accuracy with lower model complexity. On NUDT-SIRST, SCSE-Net improves mAP50:95, mAP50, and mAP75 by 3.8, 1.4, and 5.6 percentage points, respectively. Meanwhile, the number of parameters is reduced from 10.18M to 8.46M, and GFLOPs decrease from 24.8218 to 24.4769. On IRSTD-1K and IRST640, SCSE-Net achieves mAP50:95 scores of 0.399 and 0.614, respectively, demonstrating strong adaptability to different infrared small target detection scenarios.

* ## Highlights

  * **OALS**: An orientation-aware shallow structure module that preserves weak edges, contours, directional structures, and local salient responses before early downsampling.
  * **CRGM**: A context routing gated module embedded into the Transformer encoder for multi-granularity contextual modeling and target–background discrimination.
  * **AGMS**: An adaptive granularity multi-scale module that enhances cross-scale feature adaptation through multi-receptive-field branches and channel recalibration.
  * **Structure–context–scale collaboration**: SCSE-Net progressively strengthens infrared small target representation through shallow structural preservation, high-level contextual modeling, and cross-scale feature adaptation.
  * **Efficient detection**: SCSE-Net improves detection accuracy while reducing model complexity, decreasing parameters from 10.18M to 8.46M and GFLOPs from 24.8218 to 24.4769.



## Installation

### 1. Clone the repository

```bash
git clone https://github.com/mshen040123-design/SCSE-NET
cd SCSE-NET
```

### 2. Create a virtual environment

```bash
conda create -n scsenet python=3.10 -y
conda activate scsenet
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Recommended environment:

```text
Python 3.10
PyTorch 2.2.2
CUDA 12.1
TensorRT 10.11.0
```

## Dataset Preparation

Experiments are conducted on three public aerial remote sensing object detection datasets:

* **NUDT-SIRST**
* **IRSTD-1K**
* **IRST-640**

Please download the datasets from their official sources and organize them according to the paths specified in the configuration files.

A common dataset structure is:

```text
datasets/
├── NUDT-SIRST/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── IRSTD-1K/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── annotations/
│       ├── train.json
│       └── val.json
└── IRST640/
    ├── images/
    │   ├── train/
    |   ├── val/
    │   └── test/
    └── annotations/
        ├── train.json
        ├── val.json
        └── test.json
        
        
```

Please modify the dataset paths in the configuration files before training or evaluation.

## Training

Train SCSE-NET

```bash
python train.py -c configs/yaml/deim_dfine_hgnetv2_s_mg_sirst.yml
```

If only one configuration file is provided, please modify the dataset path, number of classes, category names, and training schedule according to the target dataset.

## Evaluation

Evaluate a trained model:

```bash
python train.py \
  -c configs/yaml/deim_dfine_hgnetv2_s_mg_sirst.yml \
  -r path/to/checkpoint.pth \
  --test-only
```

For other datasets, replace the configuration file and checkpoint path accordingly.

## Inference

Run inference on custom aerial images:

```bash
python tools/inference/detect/torch_inf.py \
  -c configs/yaml/deim_dfine_hgnetv2_s_mg_sirst.yml \
  -r path/to/checkpoint.pth \
  --input path/to/images \
  --output runs/inference \
  -t 0.2
```

Arguments:

```text
-c        Path to the configuration YAML file.
-r        Path to the trained model checkpoint.
--input   Input source, including a single image, video, or image folder.
--output  Directory for saving inference results.
-t        Confidence threshold for detection filtering. The default value is 0.2.
```

## Citation

If this work is useful for your research, please consider citing:

```bibtex
@misc{du2026scsenet,
  title={Structure-Context-Scale Collaborative Enhancement Network for Infrared Small Target Detection},
  author={Du, Ruiqing and Shen, Mengmeng and Xiao, Feng},
  year={2026},
  note={Manuscript under review}
}
```

## Acknowledgements

This implementation is developed for infrared small target detection research. We thank the contributors of related open-source detection frameworks, especially the DEIM training framework and D-FINE detector, as well as the providers of the NUDT-SIRST, IRSTD-1K, and IRST640 datasets.


## Contact

For questions or discussions, please contact:

```text
Mengmeng Shen
Email: mmshen04@qq.com
```