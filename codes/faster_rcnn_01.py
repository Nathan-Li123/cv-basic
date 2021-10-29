import torch
import torchvision
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torchvision.transforms.functional as TF
import numpy as np

# COCO数据集标签对照表
COCO_CLASSES = {1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}

COLORS = ['#e6194b', '#3cb44b', '#ffe119', '#0082c8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
          '#d2f53c', '#fabebe', '#008080', '#000080', '#aa6e28', '#fffac8', '#800000', '#aaffc3', '#808000',
          '#ffd8b1', '#e6beff', '#808080']

# 为每一个标签对应一种颜色，方便我们显示
LABEL_COLOR_MAP = {k: COLORS[i % len(COLORS)] for i, k in enumerate(COCO_CLASSES.keys())}

# 判断GPU设备是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def single_faster_rcnn_detection(img_path):
    origin_img = mpimg.imread(img_path)
    img = TF.to_tensor(origin_img)
    img = img.to(device)
    return model(img.unsqueeze(0))


def draw_bbox(in_path, out_path):
    labels = output[0]['labels'].cpu().detach().numpy()
    scores = output[0]['scores'].cpu().detach().numpy()
    bboxes = output[0]['boxes'].cpu().detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    origin_img = mpimg.imread(in_path)
    ax.imshow(origin_img)

    obj_index = np.argwhere(scores > 0.5).squeeze(axis=1).tolist()
    for i in obj_index:
        x1, y1, x2, y2 = bboxes[i]
        w = x2 - x1
        h = y2 - y1
        ax.add_patch(plt.Rectangle(xy=(x1, y1), width=w, height=h, fill=False, edgecolor=LABEL_COLOR_MAP[labels[i]]))
        plt.text(x1+4, y1-14, COCO_CLASSES[labels[i]], color='white', verticalalignment='top', fontsize=8,
                 bbox=dict(facecolor=LABEL_COLOR_MAP[labels[i]], edgecolor=LABEL_COLOR_MAP[labels[i]]))
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0)


if __name__ == '__main__':
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = model.to(device)
    model.eval()
    output = single_faster_rcnn_detection('./resource/input.jpg')
    draw_bbox('./resource/input.jpg', './resource/output.png')
