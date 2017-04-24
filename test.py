from v100 import *
import numpy as np
from PIL import Image
import time
from matplotlib import pyplot as plt
import uuid

img = cv2.imread('testFiles/488779608_ce29b2eda0_218_8166339@N08.xml_pic.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img_rgb = img
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
img_annotated = np.copy(img_rgb)

id = uuid.uuid1()
FaceInfo = detect_face(img_gray, img_rgb, img_annotated, True, id)
extract_features(FaceInfo, id)
LSTM_aggregate(FaceInfo)

# for i in range(0, len(FaceInfo)):
#     features = FaceInfo[i]['feature']
#     features_list = []
#     for feature in features:
#         features_list.append(feature[0].tolist())
#
#     FaceInfo[i]['feature'] = features_list
#
#     aggregated_feature = FaceInfo[i]['LSTM_aggregateed_feature']
#     FaceInfo[i]['LSTM_aggregateed_feature'] = aggregated_feature[0].tolist()

individual_estimate(FaceInfo)
group_estimation = normalME_group_estimate(FaceInfo)

# pil_im = Image.fromarray(img_annotated)
# pil_im.save('testFiles/img_tosave.jpg')
