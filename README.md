# SingleLM-Net

This repository serves as the official implementation for [Reconstructing fisheye luminance maps with a two-step network from a single low dynamic range image](https://doi.org/10.1016/j.autcon.2024.105294).

## Introduction

Implementing passive daylighting strategies is hindered by glare. While the high dynamic range (HDR) method aids in real-time glare control, its processing time and complexity are limiting. Therefore, we propose a two-step network using Generative Adversarial Networks (GANs) and a Reconstruction-Net to transform a single low dynamic range (LDR) image into a comprehensive fisheye luminance map.

<p align="center">
  <img src="./Figures/model.png"  alt="" align=center />
</p>

Our work achieves the state-of-the-art in restoring luminance maps.

Quantitative comparison of the luminance map test dataset with current methods incorporates three metrics: peak signal-to-noise ratio (PSNR) with the standard deviation, R² of daylight glare probability (DGP), and R² of vertical illuminance (E_V). All models are retrained using the luminance map dataset, and the normalization range of data processing is adjusted to ensure that the models effectively function. For all metrics, the higher they are, the better. The bold text indicates the best performance of the metric.

| Method       | PSNR↑          | R² of DGP ↑  | R² of E_V ↑ |
|--------------|----------------|--------------|--------------|
| HDRCNN  | 44.69/4.93     | 0.5987       | 0.5013       |
| Expandnet | 53.52/11.64   | 0.2563       | 0.3585       |
| HDRUNet | 58.72/9.94      | 0.8681       | 0.9112       |
| SingleLDR | 54.82/14.27   | 0.2000       | 0.2724       |
| **Ours**    | **59.24/11.97**    | **0.9054**       | **0.9243**       |

## Architecture

### LDR-GAN

To restore the absent pixel values resulting from underexposed or overexposed configurations, we introduce a low dynamic range-generative adversarial network (LDR-GAN). The generator architecture of LDR-GAN consists of multilevel scales designed to capture distinct features of the input, as informed by Pix2PixHD. The discriminator of LDR-GAN is PatchGAN.

<p align="center">
  <img src="./Figures/LDR-GAN.png"  alt="" align=center />
</p>

### Reconstruction-Net

The Reconstruction-Net is utilized to generate a high dynamic range (HDR) image, enabling the recovery of a luminance map.

<p align="center">
  <img src="./Figures/Reconstruction-Net.png"  alt="" align=center />
</p>

## Usage 

- [SingleLM-Net](#singlelm-net)
  - [Introduction](#introduction)
  - [Architecture](#architecture)
    - [LDR-GAN](#ldr-gan)
    - [Reconstruction-Net](#reconstruction-net)
  - [Usage](#usage)
    - [Dataset](#dataset)
    - [Configuration](#configuration)
    - [How to test](#how-to-test)
    - [How to train](#how-to-train)
    - [How to assess](#how-to-assess)
    - [Acknowledgment](#acknowledgment)

### Dataset

The compiled dataset was validated using an independent illuminance meter, employing the vertical illuminance (E_V) metric. The E_V values obtained through the HDR method closely match those acquired from the independent illuminance meter. You can access the luminance map dataset on [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EfNtqpM0aWJOhCImYkNUEocBjcIP40wRmOqEZbORq6x_NA?e=SMnkEY) or [Kaggle](https://www.kaggle.com/datasets/shikangwen/luminance-map-dataset). Prior to training the Reconstruction-Net, it is crucial to initially train the LDR-GAN using the LDR-GAN dataset. You can download the dataset on [OneDrive](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EWFrVCdjk7BEja3-D_MFuPUBAI_NPhf6u6yTykmJt_gY0Q?e=NIEpxj) or [Kaggle](https://www.kaggle.com/datasets/shikangwen/ldr-gan-dataset).

<p align="center">
  <img src="./Figures/Comparedwithmeter.png"  alt="" align=center />
</p>

### Configuration
To utilize the GPU capabilities of the 30 or 40 series, it is essential to install nvidia-tensorflow. However, please note that nvidia-tensorflow is exclusively compatible with Linux operating systems and python 3.8.

```bash
conda create --name singleLM python=3.8
conda activate singleLM
pip install SingleLM-Net.txt
```

### How to test

- Modify args with the `dataroot` and `pretrain_model` (you can also use the pre-trained model provided in the [pretrained model](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EU2RYKbbC-pAleOCopcO5w8BV-tmJQ5k4P5emIDBP6Dudg?e=gPA9He)) in the following command, then run

```bash
cd SingleLM-Net
conda activate singleLM
python Reconstruction_Net_Test.py --Validation_path "your test dataset" --Checkpoint_path "your pre-trained model" --save_hdr False --two_stage_network Unet --final_output_mask True --model_name model_name
```

The test results will be saved to `./Test_output/model_name`.

### How to train

- Option 1, Two-step training
  - Step 1: Training the LDR-GAN on the [LDR-GAN dataset](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EWFrVCdjk7BEja3-D_MFuPUBAI_NPhf6u6yTykmJt_gY0Q?e=NIEpxj) or using [our pre-trained model](link). If you want to use the VGG module, please download the [VGG pre-trained ckpt](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EdnoY01gNnhFvf2a5bTqfYQBHb28DuVYo1BxGl3G0q8Vjg?e=dhNyHK) first.

  ```bash
  cd SingleLM-Net
  conda activate singleLM
  python LDR_GAN_Training.py --dataroot "your training dataset" --batch_size 8 --D_lr 0.00001 --G_lr 0.00001 --vgg True --ckpt_vgg "your vgg pre-trained ckpt" --vgg_ratio 0.001  --Validation True --Validation_path "your validation dataset"  --model_name model_name
  ```

  - Step 2: Training the Reconstruction-Net on the [luminance map dataset](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EfNtqpM0aWJOhCImYkNUEocBjcIP40wRmOqEZbORq6x_NA?e=SMnkEY).

  ```bash
  python Reconstruction_Net_Training.py --batch_size 8 --learning_rate 0.0001 --vgg True --ckpt_vgg "your vgg pre-trained ckpt" --vgg_ratio 0.001 --dataroot "your training dataset" --two_stage_network Unet --Validation True --Validation_path "your validation dataset" --restore_gan True --ckpt_gan "your pre-trained LDR-GAN model" --model_name model_name
  ```

- Option 2, Directly training

  Training the SingleLM-Net on the [luminance map dataset](https://sjtueducn-my.sharepoint.com/:u:/g/personal/1063175952_sjtu_edu_cn/EfNtqpM0aWJOhCImYkNUEocBjcIP40wRmOqEZbORq6x_NA?e=SMnkEY).

  ```bash
  conda activate singleLM
  python Reconstruction_Net_Training.py --batch_size 8 --learning_rate 0.0001 --vgg True --ckpt_vgg "your vgg pre-trained ckpt" --vgg_ratio 0.001 --dataroot "your training dataset" --two_stage_network Unet --Validation True --Validation_path "your validation dataset" --restore_gan False --model_name model_name
  ```

### How to assess

To assess the model performance, calculate glare information PSNR. Ensure you have installed the `Radiance` and relevant packages before calculating the glare metric.

```bash
cd Assessment
python Calculate_metric.py --reference_HDR "your Reference dataroot" --Pred_HDR "your pred luminance map dataroot" --threads 12 --model_name model_name
```

### Acknowledgment

The code is inspired by [HDR-GAN](https://github.com/nonu116/HDR-GAN.git), [Pix2PixHD-Tensorflow](https://github.com/tiandiao123/Pix2PixHD-TensorFlow.git), and [SingleHDR](https://github.com/alex04072000/SingleHDR.git).