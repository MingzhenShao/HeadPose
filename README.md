# HeadPose

1. [Face detection](#face-detection)
2. [Datasets](#datasets)
3. [How to use our model](#train)
4. [Reference](#reference)


## Face detection
We used the face detect method proposed by Chen et al. We will give the bounding box in the list. You can also use [Microsoft face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/) to get the same result if you want to use our method in your own dataset.

## Datasets
- [300W, AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm)  
- [BIWI](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#)  
- [SASE](https://icv.tuit.ut.ee/databases/)
## Train
To train on 300W and test on AFLW2000 with bounding box margin = 0.5  
`python pose.py 0.5`


## Reference


If you find the work useful in your research please consider citing:  

```
@article{,
  author    = {Mingzhen Shao and
               Zhun Sun and
               Mete Ozay and Takayuki Okatani},
  title     = {Improving Head Pose Estimation with a Combined Loss and Bounding Box Margin Adjustment},
  journal   = {},
  volume    = {},
  year      = {2019},
  url       = {},
  archivePrefix = {},
  eprint    = {},
  timestamp = {},
  biburl    = {},
  bibsource = {}
```

