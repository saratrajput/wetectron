"""
This script computes the object proposal using the EdgeBoxes algorithm.
EdgeBoxes: https://pdollar.github.io/files/papers/ZitnickDollarECCV14edgeBoxes.pdf

author: Suraj Pattar
date: 4 April 2022
"""
import argparse
import logging
import cv2
import numpy as np

import json
import os
import pickle

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
    return parser.parse_args()


def compute_edge_boxes(img, edge_detection_obj):
    """
    Compute edge boxes with OpenCV implementation.
    https://docs.opencv.org/3.4/d4/d0d/group__ximgproc__edgeboxes.html
    """
    
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Get the edges
    edges = edge_detection_obj.detectEdges(np.float32(rgb_im)/255.0)

    # Create an orientation map
    orient_map = edge_detection_obj.computeOrientation(edges)

    # Suppress edges
    edges = edge_detection_obj.edgesNms(edges, orient_map)

    # Create edge box:
    edge_boxes = cv2.ximgproc.createEdgeBoxes()

    edge_boxes.setMaxBoxes(100)
    edge_boxes.setAlpha(0.5)
    edge_boxes.setBeta(0.5)
    prop_boxes, scores = edge_boxes.getBoundingBoxes(edges, orient_map)
    
    return prop_boxes, scores

def main(args):
    """
    Implement the main function.
    """
    input_dir = args.input_dir
    output_file = args.output_file
    im_dir = args.im_dir
    label_file = args.label_file
    
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
        box, score = compute_edge_boxes(img, edge_detection_obj)

        # Convert to dtype float32
        box = box.astype(np.float32)
        boxes.append(box)
        scores.append(score)
        indexes.append(index)
    
    # Create dictionary 
    proposal = {
    'boxes': boxes,
    'scores': scores,
    'indexes': indexes,
    }
    
    # Save pickle file
    with open(os.path.join(output_file + ".pkl"), 'wb') as handle:
        pickle.dump(proposal, handle)


if __name__=="__main__":
    args = argument_parser()
    main(args)