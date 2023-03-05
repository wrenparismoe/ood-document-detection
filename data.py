import random
import datasets
from datasets import Dataset
import pandas as pd
from datasets import load_from_disk, load_dataset
import json
from datasets import Features, Sequence, ClassLabel, Value, Array2D
from pytorch_lightning import LightningDataModule
import os

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
        train_dataset = datasets['train'].map(lambda example: encode_example(example), features=features, keep_in_memory=True, num_proc=16)
        train_dataset.set_format(type="torch",
                                 columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    else:
        train_dataset = None
    if 'validation' in datasets and is_id:
        dev_dataset = datasets['validation'].map(lambda example: encode_example(example), features=features, keep_in_memory=True, num_proc=4)
        dev_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    else:
        dev_dataset = None
    if 'test' in datasets and is_id:
        test_dataset = datasets['test'].map(lambda example: encode_example(example), features=features, keep_in_memory=True, num_proc=4)
        test_dataset.set_format(type="torch",
                                columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])
    elif 'test' in datasets and not is_id:
        test_dataset = datasets['test'].map(lambda example: encode_ood_example(example), features=ood_features, keep_in_memory=True, num_proc=4)
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
    train_df = pd.read_csv("/tmp/wpm/data/processed_train.csv")
    val_df = pd.read_csv("/tmp/wpm/data/processed_val.csv")
    test_df = pd.read_csv("/tmp/wpm/data/processed_test.csv")
    
    # Datasets map arguments: keep_in_memory=bool, num_proc=int
    
    updated_train = Dataset.from_pandas(train_df[80000:]).map(parse_json, keep_in_memory=True, num_proc=8)
    updated_val = Dataset.from_pandas(val_df).map(parse_json, keep_in_memory=True, num_proc=8)
    updated_test = Dataset.from_pandas(test_df[30000:33333]).map(parse_json, keep_in_memory=True, num_proc=8)

    datasets = {'train': updated_train, 'validation': updated_val, 'test': updated_test}
    return datasets


def load_ood():
    ood_df = pd.read_csv("/tmp/wpm/data/processed_ood.csv")
    ood_df = Dataset.from_pandas(ood_df)
    updated_ood = ood_df.map(parse_json, keep_in_memory=True, num_proc=8)
    datasets = {'test': updated_ood}
    return datasets




class DataModule(LightningDataModule):
    def __init__(self, args, train_size=None, val_size=None, test_size=None, ood_size=None, use_from_disk=True, reuse_from_disk=False, save_parquets=False, load_parquets=False):
        """
        DataModule to facilitate saving/loading of the RVL_CDIP dataset for train/val/test splits.
        """
        super().__init__()
        self.batch_size = args.batch_size
        self.train_size = train_size
        self.val_size = val_size
        if test_size is None:
            if ood_size is not None:
                self.test_size = ood_size
            else:
                self.test_size = 3000
        else:
            self.test_size = test_size
        self.ood_size = ood_size
        self.use_from_disk= use_from_disk
        self.reuse_from_disk = reuse_from_disk
        self.save_parquets = save_parquets
        self.load_parquets = load_parquets

        self.features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'label': ClassLabel(num_classes=16),
            'image_dir': Value(dtype='string'),
            'words': Sequence(feature=Value(dtype='string')),
        })
        self.ood_features = Features({
            'input_ids': Sequence(feature=Value(dtype='int64')),
            'bbox': Array2D(dtype="int64", shape=(512, 4)),
            'attention_mask': Sequence(Value(dtype='int64')),
            'token_type_ids': Sequence(Value(dtype='int64')),
            'label': ClassLabel(num_classes=1),
            'image_dir': Value(dtype='string'),
            'words': Sequence(feature=Value(dtype='string')),
        })

    def parse_json(self, example):
        json_file = example['image_dir']
        with open(json_file, 'r') as file:
            ocr_result = json.load(file)
        return ocr_result
    
    def encode_example(self, example, pad_token_box=[0, 0, 0, 0]):

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

    def encode_ood_example(self, example, pad_token_box=[0, 0, 0, 0]):

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

    def prepare_data(self):
        # preliminary steps only called on 1 GPU/TPU in distributed
        if self.reuse_from_disk:
            return

        train_df = pd.read_csv("/tmp/wpm/data/processed_train.csv")
        val_df = pd.read_csv("/tmp/wpm/data/processed_val.csv")
        test_df = pd.read_csv("/tmp/wpm/data/processed_test.csv")
        ood_df = pd.read_csv("/tmp/wpm/data/processed_ood.csv")
        
        updated_train = Dataset.from_pandas(train_df[:self.train_size]).map(self.parse_json, keep_in_memory=True, num_proc=16)
        updated_val = Dataset.from_pandas(val_df[:self.val_size]).map(self.parse_json, keep_in_memory=True, num_proc=8)
        updated_test = Dataset.from_pandas(test_df[:self.test_size]).map(self.parse_json, keep_in_memory=True, num_proc=4)
        updated_ood = Dataset.from_pandas(ood_df[:self.ood_size]).map(self.parse_json, keep_in_memory=True, num_proc=4)

        train_dataset = updated_train.map(lambda example: self.encode_example(example), features=self.features, keep_in_memory=True, num_proc=16)
        train_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])

        dev_dataset = updated_val.map(lambda example: self.encode_example(example), features=self.features, keep_in_memory=True, num_proc=8)
        dev_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])

        test_dataset = updated_test.map(lambda example: self.encode_example(example), features=self.features, keep_in_memory=True, num_proc=4)
        test_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])

        ood_dataset = updated_ood.map(lambda example: self.encode_ood_example(example), features=self.ood_features, keep_in_memory=True, num_proc=4)
        ood_dataset.set_format(type="torch", columns=['input_ids', 'bbox', 'attention_mask', 'token_type_ids', 'label'])

        # Tokenize,
        # Save processed datasets to disk (also parquet files)
        train_dataset.save_to_disk("/tmp/wpm/data/train_dataset")
        dev_dataset.save_to_disk("/tmp/wpm/data/dev_dataset")
        test_dataset.save_to_disk("/tmp/wpm/data/test_dataset")
        ood_dataset.save_to_disk("/tmp/wpm/data/ood_dataset")
        if self.save_parquets:
            train_dataset.to_parquet("/tmp/wpm/data/train_dataset.parquet")
            dev_dataset.to_parquet("/tmp/wpm/data/dev_dataset.parquet")
            test_dataset.to_parquet("/tmp/wpm/data/test_dataset.parquet")
            ood_dataset.to_parquet("/tmp/wpm/data/ood_dataset.parquet")

    def setup(self, stage=None):
        # make assignments here (val/train/test split) - called on every process in DDP
        # Load tokenized datasets from disk here
        if self.use_from_disk:
            self.train_dataset = load_from_disk("/tmp/wpm/data/train_dataset")
            self.dev_dataset = load_from_disk("/tmp/wpm/data/dev_dataset")
            self.test_dataset = load_from_disk("/tmp/wpm/data/test_dataset")
            self.ood_dataset = load_from_disk("/tmp/wpm/data/ood_dataset")
        elif self.load_parquets:
            self.train_dataset = load_dataset("tmp/wpm/data/train_dataset.parquet")
            self.dev_dataset = load_dataset("tmp/wpm/data/dev_dataset.parquet")
            self.test_dataset = load_dataset("tmp/wpm/data/test_dataset.parquet")
            self.ood_dataset = load_dataset("tmp/wpm/data/ood_dataset.parquet")

    def train_dataloader(self):
        # Train method
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)

    def val_dataloader(self):
        # evaluate methods (w/ validation and test)
        return [DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8),
                DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)]

    def test_dataloader(self):
        # prepare_ood() method (w/ validation)
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)
    
    def predict_dataloader(self):
        # evaluate_ood() method (test and ood)
        return [DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8),
                DataLoader(self.ood_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=8)]



