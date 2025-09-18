# Requirements

Python 3.7
Tensorflow 1.15
Matplotlib



## Example commands

### Training model from image-classification

pixi run python ".\src\image-classification\train.py" -p "[path to training set]" -iw 192 -ih 192 -nc 2 -bs 8 -e 2000

### Saving model from image-classification

pixi run python ".\src\image-classification\save_model.py" -mp "[path to .hdf5 model]" -on "[output model name]" -iw 192 -ih 192 -nc 2

