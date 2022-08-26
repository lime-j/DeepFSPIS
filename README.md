# DeepFSPIS
This repository contains the official implementation of "**Deep** **F**lexible **S**tructure **P**reserving **I**mage **S**moothing" [\[Link\]](https://papers.mingjia.li/DeepFSPIS.pdf), which will appear in ACM Multimedia 2022.


## Code

Currently, we have released the inference code of DeepFSPIS. (My training code is messy, will release after refactoring.) 

### Dependencies

```
pytorch >= 1.6
torchvision
numpy
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

## Code

### Component Drop

The code of our heuristic component drop can be found [here](https://github.com/lime-j/component_drop).

## Results

The results on BSDS500 test/val sets can be found [here](https://checkpoints.mingjia.li/bsds_val_test.zip).


