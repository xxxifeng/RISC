# ARCNet
## Requirement
pytorch>1.0

python=3.6
## Symbolic link the dataset to the datasets folder:
```
datasets/
        aid/
            class_1/
            class_2/
            ...
        nwpu/
            class_1/
            class_2/
            ...
```
## First: train the baseline model, fine-tune the imagenet pretrained model on remote sensing dataset:
```
    python baseline_train.py
```
The trained model parameters have been uploaded to [Baidu Netdisk](https://pan.baidu.com/s/117vQw3okXkjaA3da9WufGg?pwd=9nyj) passwd:9nyj
## Second: train the arcnet model:
```
    python train.py
```

## Output
The trained models will be saved in the model folder.
