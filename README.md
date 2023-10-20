# <div align="center">GenKIE: Robust Generative Multimodal Document Key Information Extraction</div>
<div align="center"><b>Panfeng Cao<sup>1</sup>, Ye Wang<sup>2</sup>, Qiang Zhang<sup>3</sup>, Zaiqiao Meng<sup>4</sup></b></div>

<div align="center">
<sup>1</sup>University of Michigan<br>
<sup>2</sup>National University of Defense Technology<br>
<sup>3</sup>Zhejiang University<br>
<sup>4</sup>University of Glasgow
</div>

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Datasets and Pretrained Model
We give an example of processed SROIE dataset with the question prompt to demonstrate how to use GenIE. The dataset is available at this [link](https://drive.google.com/file/d/1wioGjpXEX8MSCW68y9O_kPw-xp4CTKVk/view?usp=sharing). Our pretrained GenIE is available at this [link](https://drive.google.com/file/d/1k2nFGirVCL_8b6yTrXi75_4vSseJ7nlE/view?usp=sharing).

## Environments
```
pip install -r requirements.txt
```

## Examples
```
bash run_scripts/run_example.sh
```

## Acknowledgement
Our code is based on the work of [OFA](https://github.com/OFA-Sys/OFA). We added additional multimodal feature embedding in the encoder as described in the paper.  
