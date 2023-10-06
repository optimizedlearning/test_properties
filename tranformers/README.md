# Setup:
1. The first option is to clone the original hugging face repo using git clone:
```python3
git clone https://github.com/huggingface/transformers.git
```
Then run: 
```python3
pip install -e .
```
(this allows us to install all the packages in editable mode).
Finally, copy over the scripts in this repo to run experiments on specific tasks.

2. Copy over the directory: /projectnb/aclab/tranhp/transformers/, install packages with `pip install -e .`,  and run the appropriate scripts.
# Masked Language Modeling: 
mlm_finetune.py and mlm_pretrain.py are modified from run_mlm_no_trainer.py in https://github.com/huggingface/transformers/tree/main/examples/pytorch/language-modeling. The main difference between our scripts and the original ones is that we add a few other functions that allow us to check the state of the running model. For more details, please go through the script, every change is marked with 
```python3
#### Changed!
## Explanation for the change
<Modified block of code>
####
```
mlm_pretrain.py is also modified to load the base model without pretraining and to handle streamed datasets.

