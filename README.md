<div align="center">
  <img src="resources/mmseg-logo.png" width="600"/>
  <div>&nbsp;</div>

Documentation: <https://mmsegmentation.readthedocs.io/en/latest/>

</div>

## Installation

Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Get Started

Please see [Overview](docs/en/overview.md) for the general introduction of MMSegmentation.

Please see [user guides](https://mmsegmentation.readthedocs.io/en/latest/user_guides/index.html#) for the basic usage of MMSegmentation.
There are also [advanced tutorials](https://mmsegmentation.readthedocs.io/en/latest/advanced_guides/index.html) for in-depth understanding of mmseg design and implementation .

A Colab tutorial is also provided. You may preview the notebook [here](demo/MMSegmentation_Tutorial.ipynb) or directly [run](https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/main/demo/MMSegmentation_Tutorial.ipynb) on Colab.

To migrate from MMSegmentation 0.x, please refer to [migration](docs/en/migration).

## Tutorial

<details>
<summary>Example of Config Usage</summary>
  
  ```ruby
  _base_ = [
      '../_base_/models/upernet_beit.py', '../_base_/datasets/ade20k_640x640.py',
      '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
  ]
  
  # metainfo is a configuration that must be passed to the dataloader, otherwise it is invalid
  # palette is a display color for category at visualization
  # The palette length must be greater than or equal to the length of the classes
  METAINFO = dict(classes=('Road', 'Sidewalk', 'Construction', 'Fence', 'Pole',
                  'Traffic_Light', 'Traffic_sign', 'Nature', 'Sky','Person',
                  'Rider', 'Car', 'Background'), 
                  palette=[[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], 
                  [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], 
                  [70, 130, 180], [220, 20, 60], [0, 0, 0]])
  
  crop_size = (640, 640)
  data_preprocessor = dict(size=crop_size)
  model = dict(
      data_preprocessor=data_preprocessor,
      pretrained='pretrain/beit_base_patch16_224_pt22k_ft22k.pth',
      test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)),
      decode_head=dict(num_classes=13),
      auxiliary_head=dict(num_classes=13))
  
  optim_wrapper = dict(
      _delete_=True,
      type='OptimWrapper',
      optimizer=dict(
          type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
      constructor='LayerDecayOptimizerConstructor',
      paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))
  
  param_scheduler = [
      dict(
          type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
      dict(
          type='PolyLR',
          power=1.0,
          begin=1500,
          end=160000,
          eta_min=0.0,
          by_epoch=False,
      )
  ]
  
  # # mixed precision
  # fp16 = dict(loss_scale='dynamic')
  
  # By default, models are trained on 8 GPUs with 2 images per GPU
  train_dataloader = dict(batch_size=2)
  val_dataloader = dict(batch_size=1)
  test_dataloader = dict(batch_size=1)

  ```

</details>

<details>
<summary>Get Started</summary>

- [MMSeg overview](docs/en/overview.md)
- [MMSeg Installation](docs/en/get_started.md)
- [FAQ](docs/en/notes/faq.md)

</details>

<details>
<summary>MMSeg Basic Tutorial</summary>

- [Tutorial 1: Learn about Configs](docs/en/user_guides/1_config.md)
- [Tutorial 2: Prepare datasets](docs/en/user_guides/2_dataset_prepare.md)
- [Tutorial 3: Inference with existing models](docs/en/user_guides/3_inference.md)
- [Tutorial 4: Train and test with existing models](docs/en/user_guides/4_train_test.md)
- [Tutorial 5: Model deployment](docs/en/user_guides/5_deployment.md)
- [Deploy mmsegmentation on Jetson platform](docs/zh_cn/user_guides/deploy_jetson.md)
- [Useful Tools](docs/en/user_guides/useful_tools.md)
- [Feature Map Visualization](docs/en/user_guides/visualization_feature_map.md)
- [Visualization](docs/en/user_guides/visualization.md)

</details>

<details>
<summary>MMSeg Detail Tutorial</summary>

- [MMSeg Dataset](docs/en/advanced_guides/datasets.md)
- [MMSeg Models](docs/en/advanced_guides/models.md)
- [MMSeg Dataset Structures](docs/en/advanced_guides/structures.md)
- [MMSeg Data Transforms](docs/en/advanced_guides/transforms.md)
- [MMSeg Dataflow](docs/en/advanced_guides/data_flow.md)
- [MMSeg Training Engine](docs/en/advanced_guides/engine.md)
- [MMSeg Evaluation](docs/en/advanced_guides/evaluation.md)

</details>

<details>
<summary>MMSeg Development Tutorial</summary>

- [Add New Datasets](docs/en/advanced_guides/add_datasets.md)
- [Add New Metrics](docs/en/advanced_guides/add_metrics.md)
- [Add New Modules](docs/en/advanced_guides/add_models.md)
- [Add New Data Transforms](docs/en/advanced_guides/add_transforms.md)
- [Customize Runtime Settings](docs/en/advanced_guides/customize_runtime.md)
- [Training Tricks](docs/en/advanced_guides/training_tricks.md)
- [Contribute code to MMSeg](.github/CONTRIBUTING.md)
- [Contribute a standard dataset in projects](docs/zh_cn/advanced_guides/contribute_dataset.md)
- [NPU (HUAWEI Ascend)](docs/en/device/npu.md)
- [0.x → 1.x migration](docs/en/migration/interface.md)，[0.x → 1.x package](docs/en/migration/package.md)

</details>

## Benchmark and model zoo

Results and models are available in the [model zoo](docs/en/model_zoo.md).

<details open>
<summary>Supported backbones:</summary>

- [x] ResNet (CVPR'2016)
- [x] ResNeXt (CVPR'2017)
- [x] [HRNet (CVPR'2019)](configs/hrnet)
- [x] [ResNeSt (ArXiv'2020)](configs/resnest)
- [x] [MobileNetV2 (CVPR'2018)](configs/mobilenet_v2)
- [x] [MobileNetV3 (ICCV'2019)](configs/mobilenet_v3)
- [x] [Vision Transformer (ICLR'2021)](configs/vit)
- [x] [Swin Transformer (ICCV'2021)](configs/swin)
- [x] [Twins (NeurIPS'2021)](configs/twins)
- [x] [BEiT (ICLR'2022)](configs/beit)
- [x] [ConvNeXt (CVPR'2022)](configs/convnext)
- [x] [MAE (CVPR'2022)](configs/mae)
- [x] [PoolFormer (CVPR'2022)](configs/poolformer)
- [x] [SegNeXt (NeurIPS'2022)](configs/segnext)

</details>

<details open>
<summary>Supported methods:</summary>

- [x] [SAN (CVPR'2023)](configs/san/)
- [x] [VPD (ICCV'2023)](configs/vpd)
- [x] [DDRNet (T-ITS'2022)](configs/ddrnet)
- [x] [PIDNet (ArXiv'2022)](configs/pidnet)
- [x] [Mask2Former (CVPR'2022)](configs/mask2former)
- [x] [MaskFormer (NeurIPS'2021)](configs/maskformer)
- [x] [K-Net (NeurIPS'2021)](configs/knet)
- [x] [SegFormer (NeurIPS'2021)](configs/segformer)
- [x] [Segmenter (ICCV'2021)](configs/segmenter)
- [x] [DPT (ArXiv'2021)](configs/dpt)
- [x] [SETR (CVPR'2021)](configs/setr)
- [x] [STDC (CVPR'2021)](configs/stdc)
- [x] [BiSeNetV2 (IJCV'2021)](configs/bisenetv2)
- [x] [CGNet (TIP'2020)](configs/cgnet)
- [x] [PointRend (CVPR'2020)](configs/point_rend)
- [x] [DNLNet (ECCV'2020)](configs/dnlnet)
- [x] [OCRNet (ECCV'2020)](configs/ocrnet)
- [x] [ISANet (ArXiv'2019/IJCV'2021)](configs/isanet)
- [x] [Fast-SCNN (ArXiv'2019)](configs/fastscnn)
- [x] [FastFCN (ArXiv'2019)](configs/fastfcn)
- [x] [GCNet (ICCVW'2019/TPAMI'2020)](configs/gcnet)
- [x] [ANN (ICCV'2019)](configs/ann)
- [x] [EMANet (ICCV'2019)](configs/emanet)
- [x] [CCNet (ICCV'2019)](configs/ccnet)
- [x] [DMNet (ICCV'2019)](configs/dmnet)
- [x] [Semantic FPN (CVPR'2019)](configs/sem_fpn)
- [x] [DANet (CVPR'2019)](configs/danet)
- [x] [APCNet (CVPR'2019)](configs/apcnet)
- [x] [NonLocal Net (CVPR'2018)](configs/nonlocal_net)
- [x] [EncNet (CVPR'2018)](configs/encnet)
- [x] [DeepLabV3+ (CVPR'2018)](configs/deeplabv3plus)
- [x] [UPerNet (ECCV'2018)](configs/upernet)
- [x] [ICNet (ECCV'2018)](configs/icnet)
- [x] [PSANet (ECCV'2018)](configs/psanet)
- [x] [BiSeNetV1 (ECCV'2018)](configs/bisenetv1)
- [x] [DeepLabV3 (ArXiv'2017)](configs/deeplabv3)
- [x] [PSPNet (CVPR'2017)](configs/pspnet)
- [x] [ERFNet (T-ITS'2017)](configs/erfnet)
- [x] [UNet (MICCAI'2016/Nat. Methods'2019)](configs/unet)
- [x] [FCN (CVPR'2015/TPAMI'2017)](configs/fcn)

</details>

<details open>
<summary>Supported datasets:</summary>

- [x] [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#cityscapes)
- [x] [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-voc)
- [x] [ADE20K](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#ade20k)
- [x] [Pascal Context](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#pascal-context)
- [x] [COCO-Stuff 10k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-10k)
- [x] [COCO-Stuff 164k](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-164k)
- [x] [CHASE_DB1](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#chase-db1)
- [x] [DRIVE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#drive)
- [x] [HRF](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#hrf)
- [x] [STARE](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#stare)
- [x] [Dark Zurich](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#dark-zurich)
- [x] [Nighttime Driving](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nighttime-driving)
- [x] [LoveDA](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#loveda)
- [x] [Potsdam](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-potsdam)
- [x] [Vaihingen](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isprs-vaihingen)
- [x] [iSAID](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#isaid)
- [x] [Mapillary Vistas](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#mapillary-vistas-datasets)
- [x] [LEVIR-CD](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#levir-cd)
- [x] [BDD100K](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#bdd100K)
- [x] [NYU](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md#nyu)

</details>

Please refer to [FAQ](docs/en/notes/faq.md) for frequently asked questions.

## Projects

[Here](projects/README.md) are some implementations of SOTA models and solutions built on MMSegmentation, which are supported and maintained by community users. These projects demonstrate the best practices based on MMSegmentation for research and product development. We welcome and appreciate all the contributions to OpenMMLab ecosystem.

## Contributing

We appreciate all contributions to improve MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

## Acknowledgement

MMSegmentation is an open source project that welcome any contribution and feedback.
We wish that the toolbox and benchmark could serve the growing research
community by providing a flexible as well as standardized toolkit to reimplement existing methods
and develop their own new semantic segmentation methods.

## Citation

If you find this project useful in your research, please consider cite:

```bibtex
@misc{mmseg2020,
    title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
    author={MMSegmentation Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmsegmentation}},
    year={2020}
}
```
