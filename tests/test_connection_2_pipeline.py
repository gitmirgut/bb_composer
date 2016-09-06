import matplotlib.pyplot as plt
import numpy as np
import composer.core
import cv2
from pipeline import Pipeline
from pipeline.objects import CamParameter, Image, Filename, Positions, HivePositions, CandidateOverlay
from pipeline.pipeline import get_auto_config

pipeline = Pipeline([Filename, CamParameter],  # inputs
                    [Image, Positions, HivePositions, CandidateOverlay],  # outputs
                    **get_auto_config())
img_l = 'data/test_connection_2_pipeline/Input/Cam_0_20160715130847_631282.jpg'
img_r = 'data/test_connection_2_pipeline/Input/Cam_1_20160715130847_631282.jpg'
# Load both testing images
img_left_org = cv2.imread(img_l)
img_right_org = cv2.imread(img_r)
res_l=pipeline([img_l, 'data/test_connection_2_pipeline/Input/composer_data.npz'])
res_r=pipeline([img_r, 'data/test_connection_2_pipeline/Input/composer_data.npz'])

pts_l = res_l[HivePositions]
pts_r = res_r[HivePositions]
pts = np.vstack((pts_l, pts_r))
pts[:, [0, 1]] = pts[:, [1, 0]]
c = composer.core.Composer()
c.load_arguments('data/test_connection_2_pipeline/Input/composer_data.npz')
res = c.compose_and_mark(img_left_org,img_right_org, np.array([pts]))
cv2.imwrite('data/test_connection_2_pipeline/Output/result.jpg', res)
