"""
This script converts the bbox annotation in a coco label file to point label.
The point is the center of the bbox.

author: Suraj Pattar
date: 4 April 2022
"""
import argparse
import json
import logging
import os

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
                        help="Name of output label file")
    parser.add_argument("--label_file", type=str,
                        default="labels.json",
                        help="Optional string argument")
    return parser.parse_args()


def main(args):
    """
    Convert the bbox to point labels.
    """
    input_dir = args.input_dir
    output_file = args.output_file
    label_file = args.label_file

    # Open the input annotation file and load it.
    with open(os.path.join(input_dir, label_file)) as f:
        data = json.load(f)

    # Compute the center of the bbox and add it to the annotations.
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']

        # Compute the center of the bbox.
        xc = round(x + (w/2), 3)
        yc = round(y + (h/2), 3)

        ann['point'] = [xc, yc]
        # Delete the bbox annotation. Comment this line if you'd like to keep
        # the bbox annotations.
        del ann['bbox']

    # Write the new label file.
    with open(os.path.join(input_dir, output_file + ".json"), 'w') as f:
        json.dump(data, f)


if __name__=="__main__":
    args = argument_parser()
    main(args)
