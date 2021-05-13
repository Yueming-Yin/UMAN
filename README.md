# Universal Multi-Source Domain Adaptation

Code release for  **[Universal Multi-Source Domain Adaptation](https://arxiv.org/abs/2011.02594)** 

## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- Download datasets from https://github.com/jindongwang/transferlearning

- Download pre-trained model from https://download.pytorch.org/models/resnet50-19c8e357.pth

- Write your config file in "config.yaml"

- Train (configurations in `train-config-office31.yaml` are only for Office-31 dataset):

  `python main.py --config train-config-office31.yaml`

- Test

  `python main.py --config test-config-office31.yaml`
  
- Monitor (tensorboard required)

  `tensorboard --logdir .`

## Checkpoints

We provide the representative best cases and config files for Office-31 datasets at [Google Drive](https://drive.google.com/drive/folders/15X3VY6pYZ61ZTifkshSI4QSxVZyTKgSg?usp=sharing).

## Citation
Please cite:

```
@article{Yin2020Universal,
  author    = {Yueming Yin and
               Zhen Yang and
               Haifeng Hu and
               Xiaofu Wu},
  title     = {Universal Multi-Source Domain Adaptation},
  journal   = {CoRR},
  volume    = {abs/2011.02594},
  year      = {2020}
}
```

## Contact
- 1018010514@njupt.edu.cn (Yueming Yin)
