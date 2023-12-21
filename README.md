# TSCFormer
**Unleashing the Power of CNN and Transformer for Balanced RGB-Event Video Recognition**, Xiao Wang, Yao Rong, Shiao Wang, Yuan Chen, Zhe Wu, Bo Jiang*, Yonghong Tian, Jin Tang, 
[[arXiv]()] 

<div align="center">
<img src="https://github.com/Event-AHU/TSCFormer/blob/main/figures/firstimage.jpg" width="800">
</div>


## Abstract 
Pattern recognition based on RGB-Event data is a newly arising research topic and previous works usually learn their features using CNN or Transformer. As we know, CNN captures the local features well and the cascaded self-attention mechanisms are good at extracting the long-range global relations. It is intuitive to combine them for high-performance RGB-Event based video recognition, however, existing works fail to achieve a good balance between the accuracy and model parameters. In this work, we propose a novel RGB-Event based recognition framework termed TSCFormer, which is a relatively lightweight CNN-Transformer model. Specifically, we mainly adopt the CNN as the backbone network to first encode both RGB and Event data. Meanwhile, we initialize global tokens as the input and fuse them with RGB and Event features using the BridgeFormer module. It captures the global long-range relations well between both modalities and maintains the simplicity of the whole model architecture at the same time. The enhanced features will be projected and fused into the RGB and Event CNN blocks, respectively, in an interactive manner using F2E and F2V modules. Similar operations are conducted for other CNN blocks to achieve adaptive fusion and local-global feature enhancement under different resolutions. Finally, we concatenate these three features and feed them into the classification head for pattern recognition. Extensive experiments on two large-scale RGB-Event benchmark datasets (PokerEvent and HARDVS) fully validated the effectiveness of our proposed TSCFormer.


## A CNN-Former Framework for RGB-Event based Recognition 
<div align="center">
<img src="https://github.com/Event-AHU/TSCFormer/blob/main/figures/MM_CNNFormer_v1.jpg" width="800">  
</div>


## Environment Setting 

```shell
conda create --name tscf python=3.8 -y
conda activate tscf
conda install pytorch torchvision -c pytorch 
pip install -U openmim
mim install mmengine
mim install mmcv
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
pip install -v -e .
```

## Train & Test
```
TSCFormer:

python tools/train.py configs/recognition/tscformer/tscformer.py --seed 0 --deterministic --work-dir work_dirs/tscformer_train

python tools/test.py configs/recognition/tscformer/tscformer.py work_dirs/tscformer/best_acc_top1_epoch_xx.pth --work-dir work_dirs/tscformer_test

## Experimental Results and Visualization 
<img src="https://github.com/Event-AHU/TSCFormer/figures/featVIS_p2.jpg" width="800">


## Experimental Results and Visualization

<div align="center">
<img src="https://github.com/Event-AHU/TSCFormer/blob/main/figures/featVIS_p2.jpg" width="800">  
</div>

## Acknowledgement 
Our code is implemented based on <a href="https://github.com/open-mmlab/mmaction2">MMAction2</a>.


