import random
import datasets
from datasets import Dataset
import pandas as pd
from datasets import load_dataset
import json
from datasets import Features, Sequence, ClassLabel, Value, Array2D

datasets.logging.set_verbosity(datasets.logging.ERROR)


def load(task_name, tokenizer, max_seq_length=512, is_id=False):
    print("Loading {}".format(task_name))

    if task_name == "rvl_cdip":
        datasets = load_id()
    elif task_name == 'ood':
        datasets = load_ood()

    def encode_example(example, pad_token_box=[0, 0, 0, 0]):

        words = example['words']
        normalized_word_boxes = example['bbox']

        assert len(words) == len(normalized_word_boxes)

        token_boxes = []
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = token_boxes
        encoding['label'] = example['label']

        assert len(encoding['input_ids']) == max_seq_length
        assert len(encoding['attention_mask']) == max_seq_length
        assert len(encoding['token_type_ids']) == max_seq_length
        assert len(encoding['bbox']) == max_seq_length

        return encoding

    def encode_ood_example(example, pad_token_box=[0, 0, 0, 0]):

        words = example['words']
        normalized_word_boxes = example['bbox']

        assert len(words) == len(normalized_word_boxes)

        token_boxes = []
        for word, box in zip(words, normalized_word_boxes):
            word_tokens = tokenizer.tokenize(word)
            token_boxes.extend([box] * len(word_tokens))

        # Truncation of token_boxes
        special_tokens_count = 2
        if len(token_boxes) > max_seq_length - special_tokens_count:
            token_boxes = token_boxes[: (max_seq_length - special_tokens_count)]

        # add bounding boxes of cls + sep tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        encoding = tokenizer(' '.join(words), padding='max_length', truncation=True)
        # Padding of token_boxes up the bounding boxes to the sequence length.
        input_ids = tokenizer(' '.join(words), truncation=True)["input_ids"]
        padding_length = max_seq_length - len(input_ids)
        token_boxes += [pad_token_box] * padding_length
        encoding['bbox'] = token_boxes
        encoding['label'] = 0

        assert len(encoding['input_ids']) == max_seq_length
        assert len(encoding['attention_mask']) == max_seq_length
        assert len(encoding['token_type_ids']) == max_seq_length
        assert len(encoding['bbox']) == max_seq_length

        return encoding

    features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'label': ClassLabel(num_classes=16),
        'image_dir': Value(dtype='string'),
        'words': Sequence(feature=Value(dtype='string')),
    })

    ood_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'bbox': Array2D(dtype="int64", shape=(512, 4)),
        'attention_mask': Sequence(Value(dtype='int64')),
        'token_type_ids': Sequence(Value(dtype='int64')),
        'label': ClassLabel(num_classes=1),
        'image_dir': Value(dtype='string'),
        'words': Sequence(feature=Value(dtype='string')),
    })

    if 'train' in datasets and is_id:
        train_dataset = datasets['train'].map(lambda example: encode_example(example), features=features)
        train_dataset.set_format(type="torch",
                                 columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    else:
        train_dataset = None
    if 'validation' in datasets and is_id:
        dev_dataset = datasets['validation'].map(lambda example: encode_example(example), features=features)
        dev_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    else:
        dev_dataset = None
    if 'test' in datasets and is_id:
        test_dataset = datasets['test'].map(lambda example: encode_example(example), features=features)
        test_dataset.set_format(type="torch",
                                columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    elif 'test' in datasets and not is_id:
        test_dataset = datasets['test'].map(lambda example: encode_ood_example(example), features=ood_features)
        test_dataset.set_format(type="torch",
                                columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    else:
        test_dataset = None

    return train_dataset, dev_dataset, test_dataset


def parse_json(example):
    json_file = example['image_dir']
    with open(json_file, 'r') as file:
        ocr_result = json.load(file)
    return ocr_result


def load_id():
    train_df = pd.read_csv("data/processed_train.csv")
    val_df = pd.read_csv("data/processed_val.csv")
    test_df = pd.read_csv("data/processed_test.csv")
    # train_temp = Dataset.from_pandas(train_df.iloc[0:50])
    # val_temp = Dataset.from_pandas(val_df.iloc[0:20])
    # test_temp = Dataset.from_pandas(test_df.iloc[0:20])


    updated_train = Dataset.from_pandas(train_df[0:12000]).map(parse_json)
    updated_val = Dataset.from_pandas(val_df[0:4000]).map(parse_json)
    updated_test = Dataset.from_pandas(test_df[0:4000]).map(parse_json)

    datasets = {'train': updated_train, 'validation': updated_val, 'test': updated_test}
    return datasets


def load_ood():
    ood_df = pd.read_csv("data/processed_ood.csv")
    ood_df = Dataset.from_pandas(ood_df[0:2000])
    updated_ood = ood_df.map(parse_json)
    datasets = {'test': updated_ood}
    return datasets
