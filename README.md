# BiSeNet
BiSeNet based on pytorch 0.4.1 and python 3.6

## Dataset  
Download CamVid dataset from [Google Drive](https://drive.google.com/file/d/1KRRME_NtRG-iWOyLAb7gE-eA8fTeyzUR/view) or [Baidu Yun](https://pan.baidu.com/s/16k_hSycb2wxmN3IJPpbYig)(6xw4).

  
## Pretrained model  
Download `best_dice_loss_miou_0.655.pth` in [Google Drive](https://drive.google.com/open?id=1ulUgHwFct-vFwGCAfJ4Oa9DBlNDzm5r4) or in [Baidu Yun](https://pan.baidu.com/s/1wHyO0fJhf8j93O90Cn27tA)(6y3e) and put it in `./checkpoints`  



## Demo  
```
python demo.py
```  
### Result  
Original | GT |Predict
:-:|:-:|:-:  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/test.png" width="300" height="225" alt=""/>|<img src="https://github.com/ooooverflow/BiSeNet/blob/master/test_label.png" width="300" height="225" alt=""/>|<img src="https://github.com/ooooverflow/BiSeNet/blob/master/demo.png" width="300" height="225" alt=""/>

## Train
```
python train.py
```  
Use **tensorboard** to see the real-time loss and accuracy  
#### loss on train  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/tfboard_loss.jpg" width="1343" height="260" alt=""/>  

#### pixel precision on val  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/tfboard_precision.jpg" width="1343" height="260" alt=""/>  

#### miou on val  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/tfboard_miou.jpg" width="1343" height="260" alt=""/>  

## Test
```
python test.py
```
### Result  
class|Bicyclist|Building|Car|Pole|Fence|Pedestrian|Road|Sidewalk|SignSymbol|Sky|Tree|miou
:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:
iou | 0.61 | 0.80 |0.86|0.35|0.37|0.59|0.88|0.81|0.28|0.91|0.73|0.655

This time I train the model with **dice loss** and get better result than **cross entropy loss**. I did not use lots special training strategy, you can get much better result than this repo if using task-specific strategy.  
This repo is mainly for proving the effeciveness of the model.  
I also tried some simplified version of bisenet but it seems does not preform very well in CamVid dataset.

## Future work  
* Finish real-time segmentation with camera or pre-load video  

## Reference 
* [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/tree/master)  
* [BiSeNet-paper](https://arxiv.org/pdf/1808.00897v1.pdf)

