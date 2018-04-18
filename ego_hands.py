import os
import sys
import cv2 as cv2
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt

from config import Config
from PIL import Image
import utils
import visualize
import model as modellib
import viva

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
# Save weights
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


# TODO: make a real annotation loadre
def load_annotations(image_path, ann_dir):
    rois = list()
    with open("{}/{}".format(ann_dir, os.path.splitext(image_path)[0]) + ".txt") as f:
        content = f.readlines()
        iterContent = iter(content)
        next(iterContent)
        for roi in iterContent:
            data = roi.split()
            rois.append({"id": data[0], "x": int(data[1]),
                         "y": int(data[2]), "w": int(data[3]), "h": int(data[4])})
    return rois


class EgoDataset(utils.Dataset):
    def load_ego(self, dataset_dir, subset, num=0):
        """Load ego dataset.
        dataset_dir: The root directory of the ego dataset.
        subset: Test or valuation
        """

        # TODO: annotation and image directory structure
        ann_dir = "PATH"
        image_dir = "PATH"

        self.add_class("ego", 1, "hand")

        img_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        shuffle(img_names)
        if subset == "val":
            if num is None or num <= 0:
                print("Error reading validation samples num")
                sys.exit(-1)
            img_names = img_names[0:num]

        # Add images
        for i in range(len(img_names)):
            im = Image.open(os.path.join(image_dir, img_names[i]))
            im_size = im.size
            im.close()
            self.add_image(
                "ego", image_id=i,
                path=os.path.join(image_dir, img_names[i]),
                width=im_size[0], height=im_size[1],
                annotations=load_annotations(img_names[i], ann_dir))

    def load_mask(self, image_id):
        # TODO: make a real mask loader
        return


if __name__ == '__main__':
    """Usage for ego_hands.py
    @params:
        command     - Required : train or test 
        dataset     - Required : /path/to/viva
        model       - Required : /path/to/weights.h5
        logs        - Optional : logs directory path
        wname       - Optional : name of weights file
        num         - number of image to process
    """
    # Comment this to use every GPU available
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on EGO.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on EGO")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/viva/",
                        help='Directory of the EGO dataset')
    parser.add_argument('--model', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--wname', required=False,
                        default="viva_weights",
                        metavar="weight_file_name",
                        help="Name for weights file")
    parser.add_argument('--num', required=False,
                        default=0,
                        metavar="number_of_test_images",
                        help="Insert the number of image to test, if no number it takes all.")

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)
    if args.command == "train":
        print("Weight-name: ", args.wname)
    print("Type: ", args.type)

    image_dir = args.dataset
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    MODEL_PATH = os.path.join(ROOT_DIR, args.model)

    """Train the model"""
    if args.command == "train":
        dataset_train = EgoDataset()
        dataset_train.load_ego(image_dir, args.command)
        dataset_train.prepare()

        dataset_val = EgoDataset()
        """Change the number for get more/less image to valuate"""
        dataset_val.load_ego(image_dir, "val", 500)
        dataset_val.prepare()

        config = viva.VivaConfig()
        config.display()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        """I usually train the network starting from pre-trained MS-COCO weights"""
        model.load_weights(MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')
        """At the end save weights"""
        model.keras_model.save_weights(f"./{args.wname}.h5")

    elif args.command == "test":

        class InferenceConfig(viva.VivaConfig):
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        inference_config = InferenceConfig()

        # Create the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir=MODEL_DIR)

        image_dir = args.dataset
        img_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
        class_names = ['BG', 'leftHand_driver', 'rightHand_driver', 'leftHand_passenger', 'rightHand_passenger']
        img_names.sort()

        # Get path to saved weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        model_path = args.model

        print(f"Loading weights from {model_path}...")
        model.load_weights(model_path, by_name=True)

        num = int(args.num) if int(args.num) > 0 else len(img_names)
        viva.printProgressBar(0, num, prefix='Progress:', suffix='Complete', length=50)
        for i in range(num):
            original_image = cv2.imread(
                "{}/{}".format(
                    args.dataset, img_names[i]
                )
            )
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            results = model.detect([original_image], verbose=0)
            r = results[0]

            visualize.display_instances(f"./Predicted/{img_names[i]}", original_image, r['rois'],
                                        r['masks'], r['class_ids'], class_names, r['scores'], ax=viva.get_ax())
            viva.printProgressBar(i+1, num, prefix='Progress:', suffix='Complete', length=50)

    else:
        print("'{}' is not recognized. "
              "Use 'train', 'resume' or 'test'".format(args.command))
