# Universal Multi-Source Domain Adaptation

Code release for  **[Universal Multi-Source Domain Adaptation for Image Classification](https://www.sciencedirect.com/science/article/pii/S0031320321004192)** 

## Requirements
- python 3.6+
- PyTorch 1.0

`pip install -r requirements.txt`

## Usage

- Download datasets from https://github.com/jindongwang/transferlearning

- Generate the list of your datasets (read the comments in the code and modify the relevant parameters to use, lists we used in the Dataset_lists folder are as a reference):

  `python turn_to_list.py`

- Download pre-trained model from https://download.pytorch.org/models/resnet50-19c8e357.pth

- Write your config file in "config.yaml"

- Train (configurations in `train-config-office31.yaml` are only for Office-31 dataset):

  `python main.py --config train-config-office31.yaml`

- Test

  `python main.py --config test-config-office31.yaml`
  
- Monitor (tensorboard required)

  `tensorboard --logdir .`

## Best Cases

We provide the representative best cases and config files for Office-31 datasets at [Google Drive](https://drive.google.com/drive/folders/15X3VY6pYZ61ZTifkshSI4QSxVZyTKgSg?usp=sharing).

## Citation
Please cite:

```
@article{yin2022universal,
  title={Universal multi-Source domain adaptation for image classification},
  author={Yin, Yueming and Yang, Zhen and Hu, Haifeng and Wu, Xiaofu},
  journal={Pattern Recognition},
  volume={121},
  pages={108238},
  year={2022},
  publisher={Elsevier}
}
```
or
```
Yin, Yueming, Zhen Yang, Haifeng Hu, and Xiaofu Wu. "Universal multi-Source domain adaptation for image classification." Pattern Recognition 121 (2022): 108238.
```

## Contact
- 1018010514@njupt.edu.cn (Yueming Yin)
