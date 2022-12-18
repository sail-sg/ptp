# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import pandas as pd
import cv2
import numpy as np
from PIL import Image
import ast
import json

ann = pd.read_csv('/vg_train_val_success_compressed.tsv', sep='\t', header=None)
# 364, 190

sample = ann.iloc[-30]

im = Image.open('VG/VG_100K_2/' + sample[0].split('/')[-1])

w, h = im.size
print(w, h)

bboxs = json.loads(sample[2])
classes = ast.literal_eval(sample[3])
img = np.asarray(im)
max_char_per_line = 30
y0, dy = 10, 20
for i in range(len(bboxs)):
    cv2.rectangle(img, (bboxs[i][0], bboxs[i][1]), (bboxs[i][0]+bboxs[i][2], bboxs[i][1]+bboxs[i][3]), (0, 255, 0), 2)
    text_img = cv2.putText(img, classes[i], (bboxs[i][0], bboxs[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0))
    bboxs[i][2] = bboxs[i][0] + bboxs[i][2]
    bboxs[i][3] = bboxs[i][1] + bboxs[i][3]
    w_1 = min(int((bboxs[i][0]/2 + bboxs[i][2]/2)/w * 3), 2)
    h_1 = min(int((bboxs[i][1]/2 + bboxs[i][3]/2)/h * 3), 2)
    # print(w_1, h_1, w, h)
    block = 3*h_1 + w_1
    prompt_text = '. The block ' + str(block) + ' has a ' + classes[i] + ' .';
    print(prompt_text)

cv2.imwrite('vg_example_1.jpg', img)
