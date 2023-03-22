import pandas as pd
import json
import numpy as np
import zlib
import os


out_json = "cc3m_269_w_bbox.json" # 2681187 samples preserved

object_dict_path = "VG-SGG-dicts-vgoi6-clipped.json"
object_dict = json.load(open(object_dict_path,'r'))['label_to_idx']
# step1: encode each sample into a vector
print("{} object classes in total".format(len(object_dict)))
# print(object_dict)

objects_class_count = np.zeros(len(object_dict))


def generate_object_bbox(objects):
    object_bboxs = []
    for index, object in enumerate(objects):
        if index < 10:
            object_bboxs.append([int(coord) for coord in object['rect']])
    # print(object_bboxs)
    return object_bboxs

# step1: generate object tags for each sample
print("===step1: begin to generate object tags caption====")
sample_index = []
object_bboxs_dict = {}

for i in range(12):
    # if i > 0:
    #     break
    src_tsv = "/Data/CC3M/{}/predictions.tsv".format(i)
    metadata = pd.read_csv(src_tsv, sep='\t', header=None)
    # append boxes and indexs
    for j in range(len(metadata)):
        num_boxes = json.loads(metadata.iloc[j][1])['num_boxes']
        index = metadata.iloc[j][0]
        sample_index.append(index)
        objects = json.loads(metadata.iloc[j][1])['objects']
        object_bboxs_dict[index] = generate_object_bbox(objects)
    print("subdir {}/{} finished".format(i+1, 12))

# step2: align cc3m with own list
print("===step2: begin to align with previous caption====")
train_set = pd.read_csv('/CC3M/Train-GCC-training.tsv', sep='\t', header=None)
val_set = pd.read_csv('/CC3M/Validation-GCC-1.1.0-Validation.tsv', sep='\t', header=None)
all_set = pd.concat([train_set, val_set])
success_data = []
# count = 0
for sample in sample_index:
    # count += 1
    # if count > 10000:
    #     break
    file_name = str(zlib.crc32(all_set.iloc(0)[sample][1].encode('utf-8')) & 0xffffffff) + '.jpg'
    if sample >= len(train_set):
        img_root = "validation"
        sub_dir = str((sample - len(train_set)) // 1000)
    else:
        img_root = "train"
        sub_dir = str(sample // 1000)
    img_path = os.path.join(img_root, sub_dir, file_name)
    rel_img_path = os.path.join(img_root, sub_dir, file_name)
    success_data.append({'image': rel_img_path, 'caption': all_set.iloc(0)[sample][0], 'object': object_bboxs_dict[sample]})

print("{} samples preserved".format(len(success_data)))
# 1484208 samples preserved

# step3: 
print("===step3: merge with caption cc====")
ann = json.load(open('../metadata/cc3m/train_success_align_269.json', 'r'))

object_caption_dict = {}

success_data_preserved = []
img_paths = dict()
for i in range(len(success_data)):
    img_paths[success_data[i]['image']] = 0
    object_caption_dict[success_data[i]['image']] = success_data[i]['object']

# find the joint part
success_data_preserved = []
for i in range(len(ann)):
    if ann[i]['image'] in img_paths.keys():
        ann[i]['object'] = object_caption_dict[ann[i]['image']]
        success_data_preserved.append(ann[i])
    if i % 1000 == 0:
        print("{}/{} finished".format(i, len(ann)))

print("{} samples preserved".format(len(success_data_preserved)))


with open(out_json, 'w') as outfile:
    json.dump(success_data_preserved, outfile)