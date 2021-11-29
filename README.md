# Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data (NeurIPS 2021)

![teaser](./resources/teaser.jpg)

This repository provides the official PyTorch implementation for the following paper:

**Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data**<br>
[Liming Jiang](https://liming-jiang.com/), [Bo Dai](http://daibo.info/), [Wayne Wu](https://wywu.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
In NeurIPS 2021.<br>
[**Project Page**](https://www.mmlab-ntu.com/project/apa/index.html) | [**Paper**](https://arxiv.org/abs/2111.06849) | [**Poster**](https://liming-jiang.com/projects/APA/resources/poster.pdf) | [**Slides**](https://liming-jiang.com/projects/APA/resources/slides.pdf) | [**YouTube Demo**](https://www.youtube.com/watch?v=3Luz817WpZM)
> **Abstract:** *Generative adversarial networks (GANs) typically require ample data for training in order to synthesize high-fidelity images. Recent studies have shown that training GANs with limited data remains formidable due to discriminator overfitting, the underlying cause that impedes the generator's convergence. This paper introduces a novel strategy called Adaptive Pseudo Augmentation (APA) to encourage healthy competition between the generator and the discriminator. As an alternative method to existing approaches that rely on standard data augmentations or model regularization, APA alleviates overfitting by employing the generator itself to augment the real data distribution with generated images, which deceives the discriminator adaptively. Extensive experiments demonstrate the effectiveness of APA in improving synthesis quality in the low-data regime. We provide a theoretical analysis to examine the convergence and rationality of our new training strategy. APA is simple and effective. It can be added seamlessly to powerful contemporary GANs, such as StyleGAN2, with negligible computational cost.*

https://user-images.githubusercontent.com/27750093/141257021-4c2aa617-3274-4b82-9577-6913d19d8a20.mp4

## Updates

- [11/2021] The **code** of APA is **released**.

- [09/2021] The [paper](https://arxiv.org/abs/2111.06849) of APA is accepted by **NeurIPS 2021**.

## Requirements

* 1&ndash;8 high-end NVIDIA GPUs with at least 12 GB of memory. We have done all testing and development using 8 NVIDIA Tesla V100 PCIe 32 GB GPUs.
* CUDA toolkit 10.1 or later. Use at least version 11.1 if running on RTX 3090. We use CUDA toolkit 10.1.
* 64-bit Python 3.7 and PyTorch 1.7.1 with compatible CUDA toolkit. See [https://pytorch.org/](https://pytorch.org/) for PyTorch install instructions. Using [Anaconda](https://www.anaconda.com/) to create a new Python virtual environment is recommended.
* Run `pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3 psutil scipy tensorboard`. 

## Inference for Generating Images

**Pretrained models can be downloaded from [Google Drive](https://drive.google.com/file/d/1V8M7c5-0SMhP6i6W2mXq5UhTx2Zad6Cw/view?usp=sharing):**

| Model | Description | FID |
| :--- | :--- | :---: |
| [afhqcat5k256x256-apa.pkl](https://drive.google.com/file/d/1P9ouHIK-W8JTb6bvecfBe4c_3w6gmMJK/view?usp=sharing) | AFHQ-Cat-5k (limited itself, 256x256), trained from scratch using APA | 4.876 |
| [ffhq5k256x256-apa.pkl](https://drive.google.com/file/d/1nKMvODpZUN1CymJ5ZlFN2zEAekfhuGgJ/view?usp=sharing) | FFHQ-5k (~7% data, 256x256), trained from scratch using APA | 13.249 |
| [anime5k256x256-apa.pkl](https://drive.google.com/file/d/1EWOdieqELYmd2xRxUR4gnx7G10YI5dyP/view?usp=sharing) | Anime-5k (~2% data, 256x256), trained from scratch using APA | 13.089 |
| [cub12k256x256-apa.pkl](https://drive.google.com/file/d/1J0qactT55ofAvzddDE_xnJEY8s3vbo1_/view?usp=sharing) | CUB-12k (limited itself, 256x256), trained from scratch using APA | 12.889 |
| [ffhq70kfull256x256-apa.pkl](https://drive.google.com/file/d/1ICkGHTUepYwVfw-0DuO5xtWwDcWoIYr5/view?usp=sharing) | FFHQ-70k (full data, 256x256), trained from scratch using APA | 3.678 |
| [ffhq5k1024x1024-apa.pkl](https://drive.google.com/file/d/1pwOD8RrngVQr6UvLg2fRuvmU23BTSyqv/view?usp=sharing) | FFHQ-5k (~7% data, 1024x1024), trained from scratch using APA | 9.545 |

The downloaded models are stored as `*.pkl` files that can be referenced using local filenames:

```bash
# Generate images with the truncation of 0.7
python generate.py --outdir=out --trunc=0.7 --seeds=1000-1199 --network=/path/to/checkpoint/pkl

# Generate images without truncation
python generate.py --outdir=out --trunc=1 --seeds=1000-1199 --network=/path/to/checkpoint/pkl
```

Outputs from the above commands will be placed under `out/*.png`, controlled by `--outdir`.

## Dataset Preparation

**Our used datasets can be downloaded from their official pages:**

| Datasets | [Animal Faces-HQ Cat (AFHQ-Cat)](https://github.com/clovaai/stargan-v2#animal-faces-hq-dataset-afhq) | [Flickr-Faces-HQ (FFHQ)](https://github.com/NVlabs/ffhq-dataset) | [Danbooru2019 Portraits (Anime)](https://www.gwern.net/Crops#danbooru2019-portraits) | [Caltech-UCSD Birds-200-2011 (CUB)](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) |
| :--- | :---: | :---: | :---: | :---: |

We use [dataset_tool.py](./dataset_tool.py) to prepare the downloaded datasets (run `python dataset_tool.py --help` for more information). The datasets will be stored as uncompressed ZIP archives containing uncompressed PNG files. Alternatively, a folder containing images can also be used directly as a dataset, but doing so may lead to suboptimal performance.

For instance, the ZIP archive (a subset of 5k images with a resolution of 256 × 256) of a custom dataset can be created from its folder containing images:

```bash
python dataset_tool.py --source=/path/to/image/folder --dest=/path/to/archive.zip \
    --width=256 --height=256 --max-images=5000
```

More detailed steps can be found at [stylegan2-ada-pytorch Preparing datasets](https://github.com/NVlabs/stylegan2-ada-pytorch#preparing-datasets).

## Training New Networks

To train a new model using the proposed APA, we recommend running the following command as a starting point to achieve desirable quality in most cases:

```bash
python train.py --outdir=./experiments --gpus=8 --data=/path/to/mydataset.zip \
    --metricdata=/path/to/mydatasetfull.zip --mirror=1 \
    --cfg=auto --aug=apa --with-dataaug=true
```

In this example, the results are saved to a newly created directory `./experiments/<ID>-mydataset-auto8`, controlled by `--outdir`. The `auto8` indicates the *base configuration* that the hyperparameters were selected automatically for training on 8 GPU. The training exports network pickles (`network-snapshot-<INT>.pkl`) and example images (`fakes<INT>.png`) at regular intervals (controlled by `--snap`).

For each pickle, the training also evaluates FID (controlled by `--metrics`) and logs the resulting scores in `metric-fid50k_full.jsonl` (as well as TFEvents if TensorBoard is installed). Following [stylegan2-ada](https://github.com/NVlabs/stylegan2-ada#preparing-training-set-sweeps), it is **noteworthy** that when trained with artifically limited/amplified datasets, the quality metrics (*e.g.*, `fid50k_full`) should still be evaluated against the corresponding original full datasets. We add this missing feature in [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) with a `--metricdata` argument to specify a separate metric dataset, which can differ from the training dataset (specified by `--data`).

This training does not necessarily lead to the optimal results, which can be further customized with additional command line options:

* `--cfg` (Default: `auto`) can be changed to [other training configurations](https://github.com/NVlabs/stylegan2-ada-pytorch#training-new-networks), *e.g.*, `paper256` for the 256x256 resolution and `stylegan2` for the 1024x1024 resolution.
* `--aug` (Default: `apa`) specifies augmentation mode, which can be adjusted to `noaug` for the no augmentation mode on sufficient data or `fixed` for a fixed deception probability (controlled by `--p`).
* `--with-dataaug` (Default: `false`) controls whether to apply standard data augmentations for the discriminator inputs. This option can be set to `false` if one would like to train a model by applying APA solely, which is effective and with negligible computational cost. Setting it to `true` (following the command above) is sometimes more desirable since APA is complementary to standard data augmentations, which is very important to boost the performance further in most cases.
* `--target` (Default: 0.6) indicates the threshold for APA heuristics. Empirically, a smaller value can be chosen when one has fewer data. Besides, a larger value (*i.e.*, `--target=0.8`) is used for the Anime dataset.
* `--gamma` (Default: depends on `--cfg`) overrides R1 gamma. Different values can be tried for a new dataset.

Please refer to `python train.py --help` and [stylegan2-ada-pytorch Training new networks](https://github.com/NVlabs/stylegan2-ada-pytorch#training-new-networks) for other options.

## Evaluation Metrics

By default, [train.py](./train.py) automatically computes FID for each network pickle exported during training. We recommend inspecting `metric-fid50k_full.jsonl` (or TensorBoard) at regular intervals to monitor the training progress.

The metrics can also be computed after the training:

```bash
python calc_metrics.py --network=/path/to/checkpoint/pkl --gpus=8 \
    --metrics=fid50k_full,is50k --metricdata=/path/to/mydatasetfull.zip --mirror=1
```

The command above calculates the `fid50k_full` and `is50k` metrics for a specified checkpoint pickle file (run `python calc_metrics.py --help` and refer to [stylegan2-ada-pytorch Quality metrics](https://github.com/NVlabs/stylegan2-ada-pytorch#quality-metrics) for more information). Similarly, the metrics should be evaluated against the corresponding original full dataset.

Some metrics may have a high one-off cost when calculating them for the first time on a new dataset. Also note that the evaluation is done using a different random seed each time, so the results could slightly vary if the same metric is computed multiple times.

## Additional Features

Please refer to [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) for other usage and statistics of the codebase. Differently, the training cost of applying APA solely is negligible as opposed to ADA that spends additional time for applying external augmentations (see our [paper](https://arxiv.org/abs/2111.06849) for details).

## Results

### Effectiveness on Various Datasets

![effectonsg2](./resources/effectonsg2.jpg)

### Effectiveness Given Different Data Amounts

![ffhqdiffamount](./resources/ffhqdiffamount.jpg)

### Overfitting and Convergence Analysis

![overfitsg2apa](./resources/overfitsg2apa.jpg)

### Comparison with Other State-of-the-Art Solutions

![compare](./resources/compare.jpg)

### Higher-Resolution Examples (1024 × 1024) on FFHQ-5k (~7% data)

![1024](./resources/1024.jpg)

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@inproceedings{jiang2021DeceiveD,
  title={{Deceive D: Adaptive Pseudo Augmentation} for {GAN} Training with Limited Data},
  author={Jiang, Liming and Dai, Bo and Wu, Wayne and Loy, Chen Change},
  booktitle={NeurIPS},
  year={2021}
}
```

## Acknowledgments

The code is developed based on [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch). We appreciate the nice PyTorch implementation.

## License

Copyright (c) 2021. All rights reserved.

The code is released under the [NVIDIA Source Code License](./LICENSE.txt).
