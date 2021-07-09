# FTC: Fast-Text-Classifier
Official PyTorch implementation of FTC - Text/Non-Text image classifier [paper](https://drive.google.com/file/d/1V4OJ9k565BoNFz7PWwvSV4EIkfAhyTin/view?usp=sharing).

## Overview
PyTorch implementation of Fast-Text-Classifier (FTC).
The model effectively predicts whether an image contains text or not.
FTC predicts text/non-text based on 3x3 and 5x5 blocks.
To obtain image-level prediction, logical OR is used. 

![examples](https://user-images.githubusercontent.com/31310903/125055904-fb95f080-e0a7-11eb-8b8a-4e23adc726fd.jpg)

Fast-Text-Classifier block-level results on images selected from the TextDis
dataset, at resolutions 3x3 (odd columns) and 5x5 (even columns). The green and red
boxes mark true and false positive predictions, respectively, according to the ground
truth TextDis labels.

## Getting started
### Install dependencies
```bash
pip install -r requirements.txt
```

### Inference
The model was trained on TextDis and [Twitter1M](http://ptak.felk.cvut.cz/personal/kraliji2/twitter1m) training sets.
For ICDAR21 results, set `--thresh` to 0.647 to obtain the best F-measure score.

```bash
python test.py --dataset_path="dataset_path" --model_path="./ftc_model.pth"
```

#### Arguments
- `--dataset_path`: Evaluation dataset path.
- `--model_path`: Path to a trained model.
- `--bs, --batch_size`: Batch size (default: 1).
- `--nw, --num_workers`: Number of workers (default: 12).
- `--width`: Input image width (default: 360).
- `--height`: Input image height (default: 360). 
- `--thresh`: Text/non-text decision threshold (default: 0.5). 
- `--dataset_thresh`: Defines a minimum text area coverage of a block to label as text block.
- `--cuda`: Use cuda (default: False).
- `--mask_path`: Path to the binary text/non masks (default: "./data/Twitter1M/masks/").


### Training
```bash
python train.py --dataset_train="training_dataset_path" --dataset_test="evaluation_dataset_path"
```

#### Arguments

- `--dataset_train`: Path to the training dataset. 
- `--dataset_test`: Path to the evaluation dataset.
- `--save_path`: Path to save results and a model  (default: "./saved_models/FTC/").
- `--lr, --learning-rate`: Initial learning rate (default: 0.0001).
- `--epochs`: Number of epochs (default: 100).
- `--bs, --batch_size`: Batch size (default: 1).
- `--nw, --num_workers`: Number of workers (default: 12).
- `--width`: Input image width (default: 360).
- `--height`: Input image height (default: 360). 
- `--thresh`: Text/non-text decision threshold (default: 0.5). 
- `--dataset_thresh`: Defines a minimum text area coverage of a block to label as text block.
- `--cuda`: Use cuda (default: False).
- `--mask_path`: Path to the binary text/non masks (default: "./data/Twitter1M/masks/").


## Citation
TODO
