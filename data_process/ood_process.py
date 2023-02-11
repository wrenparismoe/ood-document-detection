import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import json


def normalize_box(box, width, height):
    return [
        int(1000 * (box[0] / width)),
        int(1000 * (box[1] / height)),
        int(1000 * (box[2] / width)),
        int(1000 * (box[3] / height)),
    ]


def apply_ocr(example):
    # get the image
    image = Image.open(example['image_dir'])

    width, height = image.size

    # apply ocr to the image
    ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
    float_cols = ocr_df.select_dtypes('float').columns
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)

    # get the words and actual (unnormalized) bounding boxes
    # words = [word for word in ocr_df.text if str(word) != 'nan'])
    words = list(ocr_df.text)
    words = [str(w) for w in words]
    coordinates = ocr_df[['left', 'top', 'width', 'height']]
    actual_boxes = []
    for idx, row in coordinates.iterrows():
        x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
        actual_box = [x, y, x + w, y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
        actual_boxes.append(actual_box)

    # normalize the bounding boxes
    boxes = []
    for box in actual_boxes:
        boxes.append(normalize_box(box, width, height))

    # add as extra columns
    assert len(words) == len(boxes)
    example['words'] = words
    example['bbox'] = boxes
    return example


ood_path = "data/ood"
ood_files = os.listdir(ood_path)

ood_images = []
for file_name in ood_files:
    img_dir = ood_path + "/" + file_name
    ood_images.append(img_dir)

ood_data = pd.DataFrame.from_dict({'image_dir': ood_images})
ood_data.to_csv("data/ood.csv", index=False)

for i, row in ood_data.iterrows():
    example = apply_ocr(row)
    file_name = "data/ocr/ood/" + example['image_dir'].split('/')[-1] + '.json'
    with open(file_name, 'w') as outfile:
        out = example.to_dict()
        json.dump(out, outfile)
