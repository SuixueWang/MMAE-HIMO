### MMAE-HIMO: A Multimodal Masked Autoencoder Fusing Histopathological Image and Multi-omics for Hepatocellular Carcinoma Survival Prediction

This is a PyTorch implementation of the [MMAE-HIMO paper]() under Linux with GPU NVIDIA A100 80GB.

### Requirements
- pytorch 1.8.0+cu111
- Pillow 9.5.0
- timm 0.3.2
- lifelines 0.27.4

### Download
- [Representative region of whole slide images]().
- [Pre-trained checkpoint]().


### Run
Download the whole slide images first, and then move the images to the path './dataset/'. 

1. Pre-training
```angular2htm
  CUDA_VISIBLE_DEVICES=0 python3 main_pretrain.py
```

2. Survival prediction
```angular2html
  CUDA_VISIBLE_DEVICES=0 python3 main_finetune.py
```
