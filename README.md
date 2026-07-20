# BCE-Net: Reliable Building Footprints Change Extraction based on Historical Map and Up-to-Date Images using Contrastive Learning
A reliable building footprint change extraction network based on historical maps and up-to-date images. It recognized the multi-kinds of instance-level change for the building combined with the latest images and historical footprints. We provided the dataset named SI-BU for further research.

## DataSets
+ The SI-BU Dataset (latest-temporal images and change states{0,1,2,3}): Download at [SI-BU-Baiduyun](https://pan.baidu.com/s/1CNdlv51cAu2tXqRHPRftMA?pwd=2024) with extract code: 2024.  It contains 3604 and 1328 sliced tiles for training and test datasets.
+ The SI-BU Dataset (bi-temporal images and change masks{0,255}): Download at [SI-BU-Baiduyun](https://pan.baidu.com/s/1kC7QEAJRSwU-KZv-sUlG7g?pwd=2024) with extract code: 2024.  It contains 3593 tiles for the training set since we deleted some incorrect samples.
+ <div align=center><img width="600" height="350" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure9.png"/></div>

+ The modified WHU-CD Dataset: Download at [WHU-CD-Baiduyun](https://pan.baidu.com/s/1fTjBomPH0gFAHUlP5k9wqQ?pwd=2024) with extract code: 2024. It contains 1260 and 690 sliced tiles same as the [Offcial Webset](http://gpcv.whu.edu.cn/data/building_dataset.html)  
<div align=center><img width="600" height="400" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure10.png"/></div>

## Abstract

Automatic and periodic recompiling of building databases with up-to-date high-resolution images has become a critical requirement for rapidly developing urban environments. However, the architecture of most existing approaches for change extraction attempts to learn features related to changes but ignores objectives related to buildings. This inevitably leads to the generation of significant pseudo-changes, due to factors such as seasonal changes in images and the inclination of building fa¸cades. To alleviate the above-mentioned problems, we developed a contrastive learning approach by validating historical building footprints against single up-to-date remotely sensed images. This contrastive learning strategy allowed us to inject the semantics of buildings into a pipeline for the detection of changes, which is achieved by increasing the distinguishability of features of buildings from those of non-buildings. In addition, to reduce the effects of inconsistencies between historical building polygons and buildings in up-to-date images, we employed a deformable convolutional neural network to learn offsets intuitively. In summary, we formulated a multi-branch building extraction method that identifies newly constructed and removed buildings, respectively. To validate our method, we conducted comparative experiments using the public Wuhan University building change detection dataset and a more practical dataset named SI-BU that we established. Our method achieved F1 scores of 93.99% and 70.74% on the above datasets, respectively. Moreover, when the data of the public dataset were divided in the same manner as in previous related studies, our method achieved an F1 score of 94.63%, which surpasses that of the state-of-the-art method. 
<div align=center><img width="500" height="550" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure1.png"></div>


## Method
<div align=center><img width="850" height="400" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure4.png"/></div>

BCE-Net consists of four parts: a pre-trained encoder for extracting robust multi-level features; multi-task segmentation branches for extraction of newly constructed, removed, and existing buildings; a DCN-based transform module for consistent adaptive adjustment of features; and a building instance-constrained contrastive learning module for discriminating feature optimization.

## Test

+ In each fresh managed GPU container/workspace, run `make setup`. See [ENVIRONMENT.md](ENVIRONMENT.md).
+ With a persistent bind-mounted workspace, the environment can be reused. Use `./scripts/run_in_env.sh python ...` or let VS Code select the repository venv automatically.
+ The original `requirements.txt` is a legacy Windows conda export and is not used by the managed-container bootstrap.
+ Build the [DCNv2](https://github.com/CharlesShang/DCNv2/tree/master) (Deformable Convolutional Networks V2)
+ Download the trained weights at [Weights-Baiduyun](https://pan.baidu.com/s/1LjhSh3ijoxzwn8dei8Z-4g) with extract code: wyxv
+ Prepare the data and run the testXX.py， we provided a detailed description in the comments

## Train on the map-ortho dataset

The custom dataset uses a before ortho image, an after footprint mask, and a
state mask with `0=background`, `1=no change`, `2=omission`, and `3=excess`.
The generated manifest contains 800 training, 100 validation, and 100 test
samples. Five masks without matching images are recorded in the audit report
and excluded.

The full training design, label-to-branch mapping, noisy-label strategy,
metric interpretation, and qualitative review checklist are documented in
[TRAINING_DESIGN.md](TRAINING_DESIGN.md).

The paper-to-code fidelity audit is documented in
[PAPER_IMPLEMENTATION_ALIGNMENT.md](PAPER_IMPLEMENTATION_ALIGNMENT.md).
For continuing the work in a new Codex session, start with
[SESSION_HANDOFF.md](SESSION_HANDOFF.md).

```bash
cd /home/work/projects/BCE-Net

./scripts/run_in_env.sh python scripts/prepare_map_ortho_manifest.py \
  --data-root /home/work/data/change_detection/building/map-ortho

./scripts/run_in_env.sh python train_bcenet_map_ortho.py \
  --manifest dataset/map_ortho_manifest.csv \
  --init-checkpoint /home/work/models/BCE-Net/checkpoint-best-whu.pth \
  --output-dir /home/work/models/BCE-Net/map-ortho-robust-v2-20260720
```

The default training profile uses generalized cross entropy, Dice loss,
reduced boundary weights, and reduced weights for secondary change instances
to limit the impact of noisy labels. It selects the best checkpoint by the
macro average of omission and excess F1 and stops when either change head
predicts no positive pixels for three consecutive epochs.

To monitor a detached training job:

```bash
./scripts/run_in_env.sh python scripts/show_bcenet_training_status.py
tail -f /home/work/models/BCE-Net/map-ortho-robust-v2-20260720/train.log
watch -n 2 nvidia-smi
```

Training curves are written to `training-curves.png`. Fixed omission/excess
validation samples are rendered before training, after the first epoch, every
five epochs, and after the final epoch under `qualitative/`.

The repository-local `training_monitor/current` link exposes the active run
inside VS Code Explorer even though checkpoints and images are stored under
`/home/work/models`.

## Results
<div align=center><img width="410" height="490" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure12.png" title="results on sibu dataset"><img width="410" height="490" src="https://github.com/liaochengcsu/BCE-Net/blob/main/pics/figure14.png" title="results on whu-cd dataset"></div>

## Reference
https://github.com/CharlesShang/DCNv2/tree/master
