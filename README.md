# Fast-RCNN models in Torch-7 format

AlexNet and VGG-16 pretrained on ImageNet and finetuned on VOC-2007 converted to
Torch-7 format.

Original models can be found here https://github.com/rbgirshick/fast-rcnn/tree/master/data

Files:

1. `imagenet/alexnet/features.t7` and `imagenet/alexnet/top.t7`: AlexNet pretrained on ImageNet
2. `imagenet/vgg/features.t7` and `imagenet/vgg/top.t7`: VGG-16 pretrained on ImageNet
3. `caffenet_fast_rcnn_iter_40000.t7` AlexNet pretrained on ImageNet and finetuned on VOC2007
4. `vgg16_fast_rcnn_iter_40000.t7` VGG-16 pretrained on ImageNet and finetuned on VOC2007

## Download link

The files are in Yandex cloud storage:

https://yadi.sk/d/R6K_3Dk9nHXn7

## Requirements

```
luarocks install cudnn
luarocks install inn
```

## Loading

Models were cleaned with `nn.Module.clearState` and `gradWeight`, `gradBias`
removed so don't forget to run `unpack()` to restore them if needed.

```lua
require 'cudnn'
require 'inn'
net = torch.load'caffenet_fast_rcnn_iter_40000.t7':unpack()
```

## Model printout

### Finetuned

#### AlexNet

```
nn.Sequential {
  (1): nn.ParallelTable {
    input
      |`-> (1): nn.Sequential {
      |      (1): cudnn.SpatialConvolution(3 -> 96, 11x11, 4,4, 5,5)
      |      (2): cudnn.ReLU
      |      (3): cudnn.SpatialMaxPooling(3,3,2,2,1,1)
      |      (4): cudnn.SpatialCrossMapLRN
      |      (5): cudnn.SpatialConvolution(96 -> 256, 5x5, 1,1, 2,2)
      |      (6): cudnn.ReLU
      |      (7): cudnn.SpatialMaxPooling(3,3,2,2,1,1)
      |      (8): cudnn.SpatialCrossMapLRN
      |      (9): cudnn.SpatialConvolution(256 -> 384, 3x3, 1,1, 1,1)
      |      (10): cudnn.ReLU
      |      (11): cudnn.SpatialConvolution(384 -> 384, 3x3, 1,1, 1,1)
      |      (12): cudnn.ReLU
      |      (13): cudnn.SpatialConvolution(384 -> 256, 3x3, 1,1, 1,1)
      |      (14): cudnn.ReLU
      |    }
      |`-> (2): nn.Identity
       ... -> output
  }
  (2): inn.ROIPooling
  (3): nn.View
  (4): nn.Linear(9216 -> 4096)
  (5): cudnn.ReLU
  (6): nn.Dropout(0.500000)
  (7): nn.Linear(4096 -> 4096)
  (8): cudnn.ReLU
  (9): nn.Dropout(0.500000)
  (10): nn.ConcatTable {
    input
      |`-> (1): nn.Linear(4096 -> 21)
      |`-> (2): nn.Linear(4096 -> 84)
       ... -> output
  }
}
```

### VGG-16

```
nn.Sequential {
  (1): nn.ParallelTable {
    input
      |`-> (1): nn.Sequential {
      |      (1): cudnn.SpatialConvolution(3 -> 64, 3x3, 1,1, 1,1)
      |      (2): cudnn.ReLU
      |      (3): cudnn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
      |      (4): cudnn.ReLU
      |      (5): cudnn.SpatialMaxPooling(2,2,2,2)
      |      (6): cudnn.SpatialConvolution(64 -> 128, 3x3, 1,1, 1,1)
      |      (7): cudnn.ReLU
      |      (8): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
      |      (9): cudnn.ReLU
      |      (10): cudnn.SpatialMaxPooling(2,2,2,2)
      |      (11): cudnn.SpatialConvolution(128 -> 256, 3x3, 1,1, 1,1)
      |      (12): cudnn.ReLU
      |      (13): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
      |      (14): cudnn.ReLU
      |      (15): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
      |      (16): cudnn.ReLU
      |      (17): cudnn.SpatialMaxPooling(2,2,2,2)
      |      (18): cudnn.SpatialConvolution(256 -> 512, 3x3, 1,1, 1,1)
      |      (19): cudnn.ReLU
      |      (20): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
      |      (21): cudnn.ReLU
      |      (22): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
      |      (23): cudnn.ReLU
      |      (24): cudnn.SpatialMaxPooling(2,2,2,2)
      |      (25): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
      |      (26): cudnn.ReLU
      |      (27): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
      |      (28): cudnn.ReLU
      |      (29): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
      |      (30): cudnn.ReLU
      |    }
      |`-> (2): nn.Identity
       ... -> output
  }
  (2): inn.ROIPooling
  (3): nn.View
  (4): nn.Linear(25088 -> 4096)
  (5): cudnn.ReLU
  (6): nn.Dropout(0.500000)
  (7): nn.Linear(4096 -> 4096)
  (8): cudnn.ReLU
  (9): nn.Dropout(0.500000)
  (10): nn.ConcatTable {
    input
      |`-> (1): nn.Linear(4096 -> 21)
      |`-> (2): nn.Linear(4096 -> 84)
       ... -> output
  }
}
```

### Imagenet pretrained

#### Alexnet features

```
nn.Sequential {
  (1): cudnn.SpatialConvolution(3 -> 96, 11x11, 4,4, 5,5)
  (2): cudnn.ReLU
  (3): cudnn.SpatialMaxPooling(3,3,2,2,1,1)
  (4): cudnn.SpatialCrossMapLRN
  (5): cudnn.SpatialConvolution(96 -> 256, 5x5, 1,1, 2,2)
  (6): cudnn.ReLU
  (7): cudnn.SpatialMaxPooling(3,3,2,2,1,1)
  (8): cudnn.SpatialCrossMapLRN
  (9): cudnn.SpatialConvolution(256 -> 384, 3x3, 1,1, 1,1)
  (10): cudnn.ReLU
  (11): cudnn.SpatialConvolution(384 -> 384, 3x3, 1,1, 1,1)
  (12): cudnn.ReLU
  (13): cudnn.SpatialConvolution(384 -> 256, 3x3, 1,1, 1,1)
  (14): cudnn.ReLU
}
```

#### Alexnet top

```
nn.Sequential {
  (1): nn.Linear(9216 -> 4096)
  (2): cudnn.ReLU
  (3): nn.Dropout(0.500000)
  (4): nn.Linear(4096 -> 4096)
  (5): cudnn.ReLU
  (6): nn.Dropout(0.500000)
}
```

#### VGG-16 features

```
nn.Sequential {
  (1): cudnn.SpatialConvolution(3 -> 64, 3x3, 1,1, 1,1)
  (2): cudnn.ReLU
  (3): cudnn.SpatialConvolution(64 -> 64, 3x3, 1,1, 1,1)
  (4): cudnn.ReLU
  (5): cudnn.SpatialMaxPooling(2,2,2,2)
  (6): cudnn.SpatialConvolution(64 -> 128, 3x3, 1,1, 1,1)
  (7): cudnn.ReLU
  (8): cudnn.SpatialConvolution(128 -> 128, 3x3, 1,1, 1,1)
  (9): cudnn.ReLU
  (10): cudnn.SpatialMaxPooling(2,2,2,2)
  (11): cudnn.SpatialConvolution(128 -> 256, 3x3, 1,1, 1,1)
  (12): cudnn.ReLU
  (13): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
  (14): cudnn.ReLU
  (15): cudnn.SpatialConvolution(256 -> 256, 3x3, 1,1, 1,1)
  (16): cudnn.ReLU
  (17): cudnn.SpatialMaxPooling(2,2,2,2)
  (18): cudnn.SpatialConvolution(256 -> 512, 3x3, 1,1, 1,1)
  (19): cudnn.ReLU
  (20): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (21): cudnn.ReLU
  (22): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (23): cudnn.ReLU
  (24): cudnn.SpatialMaxPooling(2,2,2,2)
  (25): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (26): cudnn.ReLU
  (27): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (28): cudnn.ReLU
  (29): cudnn.SpatialConvolution(512 -> 512, 3x3, 1,1, 1,1)
  (30): cudnn.ReLU
}
```

#### VGG-16 top

```
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> output]
  (1): nn.Linear(25088 -> 4096)
  (2): cudnn.ReLU
  (3): nn.Dropout(0.500000)
  (4): nn.Linear(4096 -> 4096)
  (5): cudnn.ReLU
  (6): nn.Dropout(0.500000)
}
```

## Credits

Converted by @szagoruyko 2015
