# ViT with LLN Attention

## Install requirements
```bash
$ pip install -r requirements.txt
```

## Dataset

Download Dogs vs Cats dataset from [kaggle](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data).
Place train.zip and test.zip files in the examples/vit directory.

## Train simple ViT model

Run ViT model with LLN and Softmax Attention. Change patch-size parameter to modify sequence length.

### Train baseline ViT with Softmax Attention
```
python examples/vit/main.py --attn=softmax --patch-size=7
```

### Train ViT with LLN Attention
```
python examples/vit/main.py --attn=lln --patch-size=7
```