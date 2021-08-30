This directory contains a distilled Hopenet and test tools converted to python3. 
They produce the same results as the original ones from code directory.

The reference run on [AFLW2000](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip) dataset
using the [300W-LP, alpha 1, robust to image quality]('https://drive.google.com/u/0/uc?id=1m25PrSE7g9D2q2XJVMR6IA7RaCvWSzCR&export=download') snapshot:
```bash
python --data_dir D:\blob\AFLW2000 --filename_list data\AFLW2000.txt --dataset AFLW2000 --snapshot models\hopenet_robust_alpha1.pkl
```
produces the following output: 
```bash
Test error in degrees of the model on the 2000 test images. Yaw: 9.6818, Pitch: 9.3810, Roll: 8.5526
```

The run with the distilled version produces the same output 
```bash
python --data_dir D:\blob\AFLW2000 --filename_list data\AFLW2000.txt --dataset AFLW2000 --snapshot models\hopenet_robust_alpha1.pkl
```
