# DeepFSPIS
This repository contains the official implementation of "**Deep** **F**lexible **S**tructure **P**reserving **I**mage **S**moothing" [\[Link\]](https://papers.mingjia.li/DeepFSPIS.pdf), which will appear in ACM Multimedia 2022.

![teasor](https://raw.githubusercontent.com/lime-j/DeepFSPIS/main/teaser.png)

## Code

Currently, we have released the inference code of DeepFSPIS. (My training code is messy, will release after refactoring.) 

The released checkpoints are trained with BSDS500 Train-set and the first 5000 image (in the ascending alphabetical order of their filename) from MS-COCO.

### Dependencies

```
pytorch >= 1.6
torchvision
numpy
```

### Component Drop

The code of our heuristic component drop can be found [here](https://github.com/lime-j/component_drop). 

Please install component drop first.

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


## Results

The results on BSDS500 test/val sets can be found [here](https://checkpoints.mingjia.li/bsds_val_test.zip).


