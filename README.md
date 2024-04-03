# VideoAI-Speedrun

VQA & VSR Efficiency and Runtime Evaluation

[AIS: Vision, Graphics and AI for Streaming Workshop at CVPR](https://ai4streaming-workshop.github.io/)


```
python runtime_vqa.py --frames 30 --imsize 1920 1080 --repeat 10 --fp16
```

We run the model 10 times using a `[batch, frames, 3, H, W]` input. We warmup and syncronize GPU times to make sure that we get an accurate runtime. In the example the input to the model `VQAModel` is a tensor `[1, 30, 3, 1920, 1080]` representing a clip of 30 frames a FHD.


Sample output:

```
INFO:AIS24-VQA:------> INPUT 1920x1080, 30 frames, 1 clip
INFO:AIS24-VQA:------> Average runtime on clip 30-frames  of (test_model) is : 139.535997 ms
INFO:AIS24-VQA:------> Average runtime per frame of (test_model) is : 4.651200 ms
INFO:AIS24-VQA:------> Average FPS of (test_model) is : 214.998285 FPS
INFO:AIS24-VQA:------> MACs per clip 30-frames : 397.309594881 [G]
INFO:AIS24-VQA:------> MACs per frame : 13.243653162700001 [G]
INFO:AIS24-VQA:------> #Params 2.225153 [M]
```

We report MACs. One MACs equals roughly two FLOPs.

### Rquirements

- [ptflops](https://pypi.org/project/ptflops/) --- for calculating MACs / FLOPs.