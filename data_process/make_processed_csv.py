
import pandas as pd

train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv")
test_df = pd.read_csv("data/test.csv")
ood_df = pd.read_csv("data/ood.csv")

# generate processed data csv
train_dir = []
train_label = []
for i, row in train_df.iterrows():
    file_name = "data/ocr/train/" + row['image_dir'].split('/')[-1]+'.json'
    train_dir.append(file_name)
    train_label.append(row['label'])

processed_train = pd.DataFrame.from_dict({'image_dir':train_dir,
                                          'label':train_label})
processed_train.to_csv("processed_train.csv",index = False)


val_dir = []
val_label = []
for i, row in val_df.iterrows():
    file_name = "data/ocr/val/" + row['image_dir'].split('/')[-1]+'.json'
    val_dir.append(file_name)
    val_label.append(row['label'])

processed_val = pd.DataFrame.from_dict({'image_dir':val_dir,
                                        'label':val_label})
processed_val.to_csv("processed_val.csv",index = False)

test_dir = []
test_label = []
for i, row in test_df.iterrows():
    file_name = "data/ocr/test/" + row['image_dir'].split('/')[-1] + '.json'
    test_dir.append(file_name)
    test_label.append(row['label'])

processed_test = pd.DataFrame.from_dict({'image_dir': test_dir,
                                         'label': test_label})
processed_test.to_csv("processed_test.csv", index=False)

ood_dir = []
for i, row in ood_df.iterrows():
    file_name = "data/ocr/ood/" + row['image_dir'].split('/')[-1] + '.json'
    ood_dir.append(file_name)

processed_ood = pd.DataFrame.from_dict({'image_dir': ood_dir})
processed_ood.to_csv("processed_ood.csv", index=False)
