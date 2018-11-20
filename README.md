# BiSeNet
BiSeNet based on pytorch  

## Dataset  
Download CamVid dataset from [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/tree/master/CamVid)  
Thanks for [GeorgeSeif](https://github.com/GeorgeSeif) for his great job!

  
## Pretrained model  
Download [epoch_295.pth](https://drive.google.com/open?id=1A7NKd9zyIzratOHUqbnCMIZF8HXb-wHy) and put it in `./checkpoints`


## Demo  
```
python demo.py
```  
### Result  
Original | GT |Predict
:---|:---|:--- 
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/test.png" width="300" height="225" alt=""/>|<img src="https://github.com/ooooverflow/BiSeNet/blob/master/test_label.png" width="300" height="225" alt=""/>|<img src="https://github.com/ooooverflow/BiSeNet/blob/master/demo.png" width="300" height="225" alt=""/>

## Train
```
python train.py
```  
Use **tensorboard** to see the real-time loss and accuracy  
#### loss on train  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/tfboard_loss.png" width="1343" height="260" alt=""/>  

#### pixel precision on val  
<img src="https://github.com/ooooverflow/BiSeNet/blob/master/tfboard_precision.png" width="1343" height="260" alt=""/>  

## Test
```
python test.py
```
### Result  
Method | Cropped |Resized
:---|:---|:--- 
Pixel Accuracy | 94.1 |93.2  

**Cropped** and **Resized** means two image processing method to make the input image size fixed, it seems like Cropped input images get better result.
I guess it's because cropped masks keep the original ground truth information while resized loss it.  

## Future work  
* Finish real-time segmentation with camera or pre-load video  
* Update pixel accuracy calculation algorithm  

## Reference 
* [Semantic-Segmentation-Suite](https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/tree/master)  
* [BiSeNet-paper](https://arxiv.org/pdf/1808.00897v1.pdf)

