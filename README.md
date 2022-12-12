# DeepFSPIS
This repository contains the official implementation of "**Deep** **F**lexible **S**tructure **P**reserving **I**mage **S**moothing" [\[ACM DL\]](https://dl.acm.org/doi/abs/10.1145/3503161.3547857)[\[self-hosted\]](https://papers.mingjia.li/DeepFSPIS.pdf), which appeared in ACM Multimedia 2022.

![teasor](https://raw.githubusercontent.com/lime-j/DeepFSPIS/main/teaser.png)

Demo is avaliable at [here](https://replicate.com/lime-j/deepfspis)! 


## Code

The released checkpoints are trained with BSDS500 Train-set and the first 10000 image (in the ascending alphabetical order of their filename) from MS-COCO.

### Dependencies

```
pytorch >= 1.6
torchvision
numpy
```

### Component Drop

The code of our heuristic component drop can be found [here](https://github.com/lime-j/component_drop). 

Please install component drop first.

### Training

First, download pre-computed edge maps from [link](http://checkpoints.mingjia.li/coco_edge.zip) and train adjuster as follows:

```
python train_adjuster.py --train_dir=<COCO training image dir> --edge_dir=<COCO_edge> --workdir=./train_adjuster
```

Then train smoother as follows :

```
python train_smoother.py --train_dataset_path=<COCO train image dir> --val_dataset_path=<COCO val image dir> --workdir=./train_smoother
```

### Inference 

Inference is quite simple, as the example below.

```
python batch_inference.py              \
       --save_dir=./result             \
       --batch_size=1                  \
       --checkpoints_dir=./checkpoints \
       --image_dir=./test_image        \
       --lamb=0.6,0.5,0.4  
```

You can also use our online demo at [here](https://replicate.com/lime-j/deepfspis)

## Results

The results on BSDS500 test/val sets can be found [here](https://checkpoints.mingjia.li/bsds_val_test.zip).


