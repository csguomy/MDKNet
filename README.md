# MDKNet

Virtual Classification: Modulating Domain-Specific Knowledge for Multi-domain Crowd Counting

**Testing code of MDKNet is available.**

# Datasets Preparation
Download the datasets `ShanghaiTech A`, `ShanghaiTech B`, `UCF-QNRF` and `NWPU`. 
Then generate the density maps via `gen_den_map.py`.
After that, create a folder named `JSTL_large_4_dataset`, and directly copy all the processed data in `JSTL_large_4_dataset`.

The tree of the folder should be:
```bash
`DATASET` is `SHA`, `SHB`, `QNRF_large` or `NWPU_large`.

-JSTL_large_dataset
   -den
       -test
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
       -train
            -Npy files with the name of DATASET_img_xxx.npy, which logs the info of density maps.
   -ori
       -test_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
       -train_data
            -ground_truth
                 -MAT files with the name of DATASET_img_xxx.mat, which logs the original dot annotations.
            -images
                 -JPG files with the name of DATASET_img_xxx.mat, which logs the original image files.
```

Download the pretrained hrnet model `HRNet-W40-C` from the link `https://github.com/HRNet/HRNet-Image-Classification` and put it directly in the root path of the repository.

# Test
Download the pretrained model(mdknet.pth) via Link：https://pan.baidu.com/s/1J9mzjo5l6z3TDr0bPYi-kw, Extract Password：sqbm

or

```bash
bash download_models.sh
```

And put the model into folder `./output/MSDNet_models/`

```bash
bash test.sh
```

# Citation
If you find our work useful or our work gives you any insights, please cite:
```
@article{MingyueGuoVirtualCM,
	author = {Mingyue Guo,Binghui Chen,Zhaoyi Yan,Yaowei Wang,Qixiang Ye},
	title = {Virtual Classification: Modulating Domain-Specific Knowledge for Multi-domain Crowd Counting}
}
```
