## CorrDGM: Deep Graph Matching based Dense Correspondence learning between Non-rigid Point Clouds
Official implementation of our CorrDGM.

### Instructions
This code has been tested on Nvidia GeForce RTX 3090 with 
- Python 3.8.10 
- PyTorch 1.10.0
- CUDA 11.1 

To create a virtual environment and install the required dependences please run:
```shell
conda create --name CorrDGM python=3.8
conda activate CorrDGM
pip install -r requirements.txt
```

### Usage

#### Dataset
Please follow the processing way as described in the paper to prepare the data and set the file path.


```python
class SurrealTrain(Dataset):    (in data.py)
    self.file_name = 'your/data/path'    

class SHRECTest(Dataset):   (in data.py)
    self.file_path = 'your/data/path'
```

#### Train
To train the network on SURREAL dataset, run:
```shell
python train.py --cfg experiments/train_CorrDGM.yaml
```

#### Test
To test on SHREC dataset, run:
```shell
python eval.py --cfg experiments/test_CorrDGM.yaml
```

### Acknowledgments
In this project we use (parts of) the official implementations of the following works: 
- [CorrNet3D](https://github.com/ZENGYIMING-EAMON/CorrNet3D)
- [RGM](https://github.com/fukexue/RGM)
- [DGCNN](https://github.com/WangYueFt/dgcnn)

 We thank the respective authors for open sourcing their methods.