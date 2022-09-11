"""# libraries"""

import warnings
#warnings.filterwarnings("ignore")

# import some common libraries
import numpy as np
import os, json, cv2, random
import pandas as pd
from PIL import Image

import torch, detectron2

# setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F

"""# versions"""

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)
print("detectron2:", detectron2.__version__)

"""# paths"""

drive_path = '/gdrive/MyDrive'
test_data_path = '/other/carbon_influence'

"""# image

## object detection with detectron2
"""

detectron2_classes_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

im = cv2.imread(drive_path + test_data_path + '/test.png')
cv2_imshow(im)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# find a model from detectron2's model zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])

"""## scene recognition with places365"""

# the architecture to use
arch = 'resnet18'

# load the pre-trained weights
model_file = '%s_places365.pth.tar' % arch
if not os.access(model_file, os.W_OK):
    weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
    os.system('wget ' + weight_url)

model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()


# load the image transformer
centre_crop = trn.Compose([
        trn.Resize((256,256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# load the class label
file_name = 'categories_places365.txt'
if not os.access(file_name, os.W_OK):
    synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
    os.system('wget ' + synset_url)
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

# load the test image
img_name = drive_path + test_data_path + '/test.png'

img = Image.open(img_name)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('{} prediction on {}'.format(arch,img_name))
# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

"""# video"""

capture = cv2.VideoCapture(drive_path + test_data_path + '/video.mp4')
c=0
while capture.isOpened():
    r, f = capture.read()
    if r == False:
        break
    if c / 100 == c // 100:
        print('frame_' + str(c))
        cv2_imshow(f)
    c+=1

capture.release()
cv2.destroyAllWindows()

print(f'number of frames: {c}')

"""## test frame analysis"""

capture = cv2.VideoCapture(drive_path + test_data_path + '/video.mp4')
test_frame_no = 1600
capture.set(1,test_frame_no)
test_r, test_frame = capture.read()
capture.release()
cv2.destroyAllWindows()

cv2_imshow(test_frame)

outputs = predictor(test_frame)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

v = Visualizer(test_frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])

img = Image.fromarray(test_frame)
input_img = V(centre_crop(img).unsqueeze(0))

# forward pass
logit = model.forward(input_img)
h_x = F.softmax(logit, 1).data.squeeze()
probs, idx = h_x.sort(0, True)

print('{} prediction on {}'.format(arch,img_name))
# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

"""## carbon score"""

object_df = pd.read_csv(drive_path + test_data_path + '/objects.csv', delimiter=';')

scene_df = pd.read_csv(drive_path + test_data_path + '/scenes.csv', delimiter=';')
scene_df['Place'] = scene_df['Place'].apply(lambda x: x.strip())

capture = cv2.VideoCapture(drive_path + test_data_path + '/video.mp4')
c=0

total_objects_carbon_score = 0
total_scenes_carbon_score = 0

while capture.isOpened():

    r, f = capture.read()

    if r == False:
        break

    if c / 100 == c // 100:
        print('\n')
        print('frame_'+str(c)) # frame number

        # frame object detection and carbon score
        outputs = predictor(f)
        print('object detection:')

        objects_outs = outputs["instances"].pred_classes.cpu().numpy()
        objects_scores = outputs["instances"].scores.cpu().numpy()
        objects_carbon_score = 0

        for i in range(len(objects_outs)):
          object_name = detectron2_classes_list[objects_outs[i]]
          print('{:.3f} -> {}'.format(objects_scores[i], object_name))
          objects_carbon_score += object_df[object_df['Object']==object_name]['Score'].values[0]

        print(f'object detection carbon influence score: {objects_carbon_score}')
        total_objects_carbon_score += objects_carbon_score

        # frame scene prediction and carbon score
        img = Image.fromarray(f)
        input_img = V(centre_crop(img).unsqueeze(0))

        # forward pass
        logit = model.forward(input_img)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)
        scenes_carbon_score = 0

        print('scene prediction:')
        # output the prediction
        for i in range(0, 3):
          scene_name = classes[idx[i]]
          print('{:.3f} -> {}'.format(probs[i], scene_name))
          scenes_carbon_score += scene_df[scene_df['Place']==scene_name]['Score'].values[0]

        print(f'scene prediction carbon influence score: {scenes_carbon_score}')
        total_scenes_carbon_score += scenes_carbon_score

    c+=1

capture.release()
cv2.destroyAllWindows()

print('\n')
print(f'total object detection carbon influence score: {total_objects_carbon_score}')
print(f'total scene prediction carbon influence score: {total_scenes_carbon_score}')