# CSE547 Project 

Code for CSE547 Project, 2023

## Requirements
* [PyTorch](http://pytorch.org/)
* [Transformers](https://github.com/huggingface/transformers)
* datasets
* wandb
* tqdm
* scikit-learn

## Dataset

* RVL-CDIP Dataset (See "data/readme.txt" for detailed data descriptions)

## Training and Evaluation

Run the model with the following commands: 

```bash
>> python run.py --task_name "rvl-cdip" --loss "margin"
```

The loss can take "margin" or "self", which means using margin-based or self-supervised contrastive loss respectively.
