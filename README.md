# Neural Semantic Parsing with Semantic Anchors as Intermediate Supervision 

Code for the AAAI 2023 paper [**"Unveiling the Black Box of PLMs with Semantic Anchors: Towards Interpretable Neural Semantic Parsing"**](https://ojs.aaai.org/index.php/AAAI/article/view/26572). 

## Prerequisites

### Dependencies

Below are the Python packages our implementation relies on. 

```python
torch                       1.9.0+cu111
torchaudio                  0.9.0
torchvision                 0.10.0+cu111
transformers                4.16.2
sqlite                      3.31.1
babel                       2.9.1
```

### Datasets

The data we used in the experiments are respectively from: Overnight ([link](https://github.com/rhythmcao/semantic-parsing-dual/tree/master/data/overnight)), KQA Pro ([link](https://github.com/shijx12/KQAPro_Baselines)) and WikiSQL ([link](https://github.com/salesforce/WikiSQL/)). The preprocessed version is available [here](https://www.dropbox.com/sh/y77goxkoxvl8wsh/AACs3X6LwiPkcvm9RBrBNSkBa?dl=0). Please refer to their original release license for proper usage. 

```bash
# Extract the compressed files into the pre-specified locations
# e.g. Overnight dataset should appear in: "./data/overnight/data"
tar -zxvf data.tar.gz 
```

### PLM Checkpoints

In case Huggingface Hub may update the model checkpoints over time, we also upload the initial PLM checkpoints (without any fine-tuning) used in our experiments [here](https://www.dropbox.com/sh/y77goxkoxvl8wsh/AACs3X6LwiPkcvm9RBrBNSkBa?dl=0).

## Usage

For running BART and T5 we prepared two separate folders "`./bart/`" and "`./t5/`" that can be regarded as Python sub-modules. Below we give a tutorial for **running BART models with Overnight dataset** but you may simply replace the "`bart`" with "`t5`" for running T5,  and modify "`overnight`" into  "`kqapro`" or "`wikisql`" for running on the other 2 datasets. For running the baselines models, you may simply remove the "`--customized`" and  "`--hybrid`" flags.

### Preprocess

```bash
python -m bart.preprocess \
--input_dir ./data/overnight/data/ \
--output_dir ./exp_files/overnight/ \
--config ./data/overnight/config.py \
--model_name_or_path ./bart-base \
--customized --supervision_form hybrid
```

### Training 

```bash
# For multi-GPU training
python -m torch.distributed.run \
--rdzv_backend=c10d 
--nproc_per_node=8 \
-m bart.train \
--input_dir ./exp_files/overnight/ \
--output_dir ./exp_results/overnight/ \
--config ./data/overnight/config.py \
--model_name_or_path ./bart-base/ \ 
--batch_size 64 \
--num_train_epochs 50 \
--customized --hybrid 

# For single-GPU training
python -m bart.train \
--input_dir ./exp_files/overnight/ \
--output_dir ./exp_results/overnight/ \
--config ./data/overnight/config.py \
--model_name_or_path ./bart-base/ \
--batch_size 64 \
--num_train_epochs 50 \
--customized --hybrid 
```

### Inference

```bash
# Please run inference with single GPU only 
python -m bart.inference \
--input_dir ./exp_files/overnight/ \
--output_dir ./exp_results/overnight/ \
--model_name_or_path ./bart-base/ \
--ckpt ./exp_results/overnight/checkpoint-best/ \
--config ./data/overnight/config.py \
--customized --hybrid
```

