"""
Marcos Conde, 2024

Video Quality Assessment Challenge at AIS2024 CVPR

AIS: Vision, Graphics and AI for Streaming CVPR 2024 Workshop
"""

import os
import torch
import time
import pathlib
import logging
import argparse
import numpy as np
import importlib
import sys
import datetime
import logging

import torch.nn.functional as F
import torch_tensorrt
import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import clogger
from model_summary import get_model_flops
from ptflops import get_model_complexity_info


class VQAModel(nn.Module):
    """
    Dummy VQA model.
    """
    def __init__(self, num_frames=60, height=1080, width=1920):
        super(VQAModel, self).__init__()
        # Initialize MobileNet
        self.mobilenet = models.mobilenet_v2(pretrained=True).features
        for param in self.mobilenet.parameters():
            param.requires_grad = False  # Freeze MobileNet parameters
        
        # Adaptive pooling to handle varying sizes
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Placeholder for the feature dimension. This needs to be adjusted based on the output of MobileNet.
        # For MobileNetV2, the feature dimension is 1280.
        feature_dim = 1280
        
        # Linear layer for quality prediction
        self.fc = nn.Linear(feature_dim, 1)  # Predicting a single score
        
    def forward(self, x):
        # x shape: [batch, frames, 3, H, W]
        batch_size, num_frames, C, H, W = x.shape
        
        # Process each frame individually
        x = x.view(batch_size * num_frames, C, H, W)  # Reshape for processing by MobileNet
        x = self.mobilenet(x) # [batch_size * num_frames, 1280, 60, 34]
        
        # Apply adaptive pooling
        x = self.pooling(x) # torch.Size([batch_size * num_frames, 1280, 1, 1])

        # Reshape back to (batch, frames, feature_dim)
        x = x.view(batch_size, num_frames, -1) # torch.Size([1, 30, 1280])
        
        # Average the features across the frames -- Simple feature aggregation
        x = torch.mean(x, dim=1) # torch.Size([1, 1280])
        
        # Predict the quality score
        x = self.fc(x)
        return x
    


def main(args):

    """
    SETUP LOGGER
    """
    clogger.logger_info("AIS24-VQA", log_path=os.path.join(args.save_dir, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("AIS24-VQA")

    """
    BASIC SETTINGS
    """
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    """
    LOAD MODEL
    """
    model = VQAModel()
    model = model.to(device)
    model.eval()
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info(f"Results of {args.submission_id}")
    logger.info('Params number: {}'.format(number_parameters))
            
    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    """
    TESTING
    """
    assert args.frames > 0
    input_dim = (1, args.frames, 3, args.imsize[0], args.imsize[1])
    input_data = torch.randn(input_dim).to(device)
    logger.info('Input resolution: {}'.format(input_data.shape))

    if args.fp16:
        input_data = input_data.half()
        model = model.half()
        if args.trt:
            model = torch_tensorrt.compile(model, inputs= [torch_tensorrt.Input(input_dim, dtype=torch_tensorrt.dtype.half)], enabled_precisions= {torch_tensorrt.dtype.half})

    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(3):
            _ = model(input_data)
            
    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(args.repeat)):       
            start.record()
            _ = model(input_data)
            end.record()

            torch.cuda.synchronize()
              
            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        logger.info('------> Average runtime of ({}) is : {:.6f} ms'.format(args.submission_id, ave_runtime))
        
        if not args.trt:
            input_dim    = (args.frames, 3, args.imsize[0], args.imsize[1])
            desired_macs = 2 * args.imsize[0] * args.imsize[1]
            macs, params = get_model_complexity_info(model, input_dim, as_strings=False,
                                           print_per_layer_stat=False, verbose=False)
            
            # one MACs equals roughly two FLOPs
            # macs2 = get_model_flops(model, input_dim, print_per_layer_stat=False)
            # macs2 = macs2 / 10 ** 9

            logger.info("{:>16s} : {:<.4f} ".format("MACs", macs))
            logger.info("{:>16s} : {:<.4f} [G]".format("MACs", macs/ 10 ** 9))

            num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
            num_parameters = num_parameters / 10 ** 6
            logger.info("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))


        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--submission-id", type=str, default="test_model")
    
    # specify dirs
    parser.add_argument("--save-dir", type=str, default="submissions/")
    
    # specify test case
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=30, help="Number of frames. Default 30FPS.")
    parser.add_argument("--imsize", type=int, nargs="+", default=[1920, 1080], help="Frame resolution.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trt", action="store_true")
    args = parser.parse_args()

    main(args)