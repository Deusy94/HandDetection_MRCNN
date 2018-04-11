import os
import cv2 as cv2
import numpy as np

def load_annotation(image_path, dir):
    rois = list()
    with open("{}/{}".format(dir, os.path.splitext(image_path)[0]) + ".txt") as f:
        content = f.readlines()
        iterContent = iter(content)
        next(iterContent)
        for roi in iterContent:
            data = roi.split()
            rois.append({"id": data[0], "x": int(data[1]),
                         "y": int(data[2]), "w": int(data[3]), "h": int(data[4])})
    return rois

if __name__ == '__main__':
    image_dir = "C:/Users/deusy/Downloads/LISA_HD_Static/detectiondata/train/pos"
    ann_dir = "C:/Users/deusy/Downloads/LISA_HD_Static/detectiondata/train/posGt"
    img_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]
    rois = load_annotation(img_names[0], ann_dir)
    img = cv2.imread(f"{image_dir}/{img_names[0]}")
    crops = []
    for i in rois:
        crops.append(img[i['y']:i['y']+i['h'], i['x']:i['x']+i['w']])
    otsu = []
    for i in range(len(crops)):
        tmp = np.zeros((crops[i].shape[0]*crops[i].shape[1], 3))
        for j in range(3):
            tmp[..., j] = np.reshape(crops[i][:, :, j], (-1, 1))[..., 0]
        ret = cv2.kmeans(tmp.astype(np.float32), 3, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 4, 2.0),
                         attempts=4, flags=cv2.KMEANS_RANDOM_CENTERS)
        print(ret)
        classified = np.reshape(ret[1], (crops[i].shape[0], crops[i].shape[1]))*255
        cv2.imshow('ciao', classified.astype(np.uint8))
        cv2.imshow('img', crops[i])
        cv2.waitKey(0)
        print("Fine prima. \n \n \n \n")
        classified = []
    '''
    for i in range(len(otsu)):
        cv2.imshow(f"{i}", otsu[i])
    cv2.waitKey(0)
    '''