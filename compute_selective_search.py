"""
This script computes the object proposal using the EdgeBoxes algorithm.
EdgeBoxes: https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf

author: Suraj Pattar
date: 4 April 2022
"""
import argparse
import json
import logging
import os
import pickle

import cv2
import numpy as np

# Initialize logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO) # Add filename='example.log' for logging to file


def argument_parser():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dir", type=str,
                        help="Input directory")
    parser.add_argument("output_file", type=str,
                        default="data",
                        help="Name of output pickle file")
    parser.add_argument("--im_dir", type=str,
                        default="data",
                        help="Name of image directory")
    parser.add_argument("--label_file", type=str,
                        default="labels.json",
                        help="Optional string argument")
    parser.add_argument("--strategy", type=str,
                        default="color",
                        help="Strategy for selective search. 'color' or 'all'")
    return parser.parse_args()


def compute_selective_search_proposals(img, strategy):
    # Instantiate selective search
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()

    # Convert image from BGR to RGB
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    ss.addImage(rgb_im)
    gs = cv2.ximgproc.segmentation.createGraphSegmentation()

    ss.addGraphSegmentation(gs)

    # Create strategy using color similarity
    if strategy == "color":
        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        ss.addStrategy(strategy_color)
    elif strategy == "all":
        # Create strategy using all similarities (size, color, fill, texture)
        strategy_color = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyColor()
        strategy_fill = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyFill()
        strategy_size = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategySize()
        strategy_texture = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyTexture()
        strategy_multiple = cv2.ximgproc.segmentation.createSelectiveSearchSegmentationStrategyMultiple(
            strategy_color, strategy_fill, strategy_size, strategy_texture)
        ss.addStrategy(strategy_multiple)

    return ss.process()


def main(args):
    """
    Initiate and compute edge boxes in a loop for all the images in path.
    """
    input_dir = args.input_dir
    output_file = args.output_file
    im_dir = args.im_dir
    label_file = args.label_file
    strategy = args.strategy
    
    # Model file for edge proposal    
    model_path = "proposal/model.yml.gz"
    # Initiate edge detection
    edge_detection_obj = cv2.ximgproc.createStructuredEdgeDetection(model_path)
    
    # Load annotation file
    with open(os.path.join(input_dir, label_file)) as f:
        ann_data = json.load(f)
        
    # Initialize boxes, scores and indexes
    boxes = []
    scores = []
    indexes = []

    for i in range(len(ann_data['images'])):
        # Image id
        index = ann_data['images'][i]['id']
        # Image name
        file_name = ann_data['images'][i]['file_name']
        # Read image
        img = cv2.imread(os.path.join(input_dir, im_dir, file_name))

        # Obtain bounding boxes and scores
        box = compute_selective_search_proposals(img, strategy)

        # Convert to dtype float32
        box = box.astype(np.float32)
        boxes.append(box)
        indexes.append(index)
    
    # Create dictionary 
    proposal = {
    'boxes': boxes,
    'indexes': indexes,
    }
    
    # Save pickle file
    with open(os.path.join(output_file + ".pkl"), 'wb') as handle:
        pickle.dump(proposal, handle)


if __name__=="__main__":
    args = argument_parser()
    main(args)
