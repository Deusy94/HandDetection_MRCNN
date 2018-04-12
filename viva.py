"""
Mask R-CNN
Configurations and data loading code for VIVA Hand Dataset.

------------------------------------------------------------

Usage:

    # Train a new model starting from pre-trained COCO weights
    python3 viva.py train --dataset=/path/to/viva/ --model=coco_weights

    # Train a new model starting from ImageNet weights
    python3 viva.py train --dataset=/path/to/viva/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 viva.py train --dataset=/path/to/viva/ --model=/path/to/weights.h5

    # Perform inference over dataset
    python3 viva.py test --dataset=/path/to/dataset/ --model=/path/to/weights.ht

"""

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

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
# Save weights
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VivaConfig(Config):
    """Configuration for training on VIVA dataset.
    Derives from the base Config class and overrides values specific
    to the VIVA dataset.
    """
    # Give the configuration a recognizable name
    NAME = "viva"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # VIVA has 4 classes

    IMAGE_MIN_DIM = 320
    IMAGE_MAX_DIM = 320

    # Use smaller anchors because our image and objects are small
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 200

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5

############################################################
#  Dataset
############################################################


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


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


def submission(name, result):
    out_file = 'MaskRCNN.txt'
    for i in range(len(result['rois'])):
        bbGt = "{} {} {} {} {} {} -1 -1 -1\n".format(name, result['rois'][i][1], result['rois'][i][0],
                                                     result['rois'][i][3] - result['rois'][i][1],
                                                     result['rois'][i][2] - result['rois'][i][0],
                                                     result['scores'][i])
        with open(out_file, "a") as myfile:
            myfile.write(bbGt)


def printProgressBar (iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


class VivaDataset(utils.Dataset):
    def load_viva(self, dataset_dir, subset, num=0):
        """Load a subset of the VIVA dataset.
        dataset_dir: The root directory of the VIVA dataset.
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        """

        ann_dir = "{}/detectiondata/train/posGt".format(dataset_dir)
        image_dir = "{}/detectiondata/train/pos".format(dataset_dir)

        self.add_class("viva", 1, "leftHand_driver")
        self.add_class("viva", 2, "rightHand_driver")
        self.add_class("viva", 3, "leftHand_passenger")
        self.add_class("viva", 4, "rightHand_passenger")

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
                "viva", image_id=i,
                path=os.path.join(image_dir, img_names[i]),
                width=im_size[0], height=im_size[1],
                annotations=load_annotations(img_names[i], ann_dir))

    def load_mask(self, image_id):
        # Have to load a mask from Bounding Box
        info = self.image_info[image_id]
        hands = info['annotations']
        count = len(hands)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, val in enumerate(hands):
            mask[:, :, i:i + 1] = cv2.rectangle(mask[:, :, i:i + 1].copy(),
                                                (val['x'], val['y']),
                                                (val['x'] + val['w'], val['y'] + val['h']),
                                                1, -1)
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count - 2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s['id']) for s in hands])
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return a link to the image in the VIVA"""
        return self.image_info[image_id]


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import argparse
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on VIVA.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' on VIVA")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/viva/",
                        help='Directory of the VIVA dataset')
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

    image_dir = args.dataset
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    # Local path to trained weights file
    MODEL_PATH = os.path.join(ROOT_DIR, args.model)

    if args.command == "train":
        dataset_train = VivaDataset()
        dataset_train.load_viva(image_dir, args.command)
        dataset_train.prepare()

        dataset_val = VivaDataset()
        dataset_val.load_viva(image_dir, "val", 500)
        dataset_val.prepare()

        config = VivaConfig()
        config.display()

        # Create model in training mode
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
        model.load_weights(MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

        # Training - Stage 1
        print("Training network heads")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='heads')

        model.keras_model.save_weights(f"./first{args.wname}.h5")

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE,
                    epochs=120,
                    layers='4+')

        model.keras_model.save_weights(f"./second_{args.wname}.h5")

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train(dataset_train, dataset_val,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=160,
                    layers='all')

        model.keras_model.save_weights(f"./final_{args.wname}.h5")

    elif args.command == "test":
        class InferenceConfig(VivaConfig):
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
        # Either set a specific path or find last trained weights
        # model_path = os.path.join(ROOT_DIR, ".h5 file name here")
        model_path = args.model

        print("Loading weights from ", model_path)
        model.load_weights(model_path, by_name=True)

        num = int(args.num) if int(args.num) > 0 else len(img_names)
        printProgressBar(0, num, prefix='Progress:', suffix='Complete', length=50)
        for i in range(num):
            original_image = cv2.imread(
                "{}/{}".format(
                    args.dataset, img_names[i]
                )
            )
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            results = model.detect([original_image], verbose=0)
            r = results[0]
            submission(img_names[i], r)
            '''
            visualize.display_instances(f"./Predicted/{img_names[i]}", original_image, r['rois'], r['masks'], r['class_ids'],
                                        class_names, r['scores'], ax=get_ax())
            if i == int((num-1) * 0.25):
                print("Fatto il 25%.")
            elif i == int((num-1) * 0.5):
                print("Fatto il 50%.")
            elif i == int((num-1) * 0.75):
                print("Fatto il 75%.")
            elif i == int(num-1):
                print("Fatto il 100%.")
            '''
            printProgressBar(i+1, num, prefix='Progress:', suffix='Complete', length=50)

    else:
        print("'{}' is not recognized. "
              "Use 'train', 'resume' or 'test'".format(args.command))
