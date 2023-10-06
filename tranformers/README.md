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
# Run the script:
One example run of mlm_finetune.py is as follows:
```python3
python mlm_finetune.py   --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --model_name_or_path bert-base-cased\
    --output_dir /projectnb/aclab/tranhp/transformers/finetune_With_wiki --num_train_epochs 5 --checkpointing_steps epoch \
    --name finetuneonWiki --data_dir /projectnb/aclab/tranhp/wiki --with_tracking --report_to wandb
```
Example run of mlm_pretrain.py where we pretrain bert on c4 dataset:
```python3
python mlm_pretrain.py     --dataset_name c4  --dataset_config_name en --path /projectnb/aclab/datasets/c4/en   \
    --model_name_or_path bert-base-cased     \
--output_dir /projectnb/aclab/tranhp/transformers/examples/pytorch/language-modeling/c4Pretrained --num_train_epochs 50  \
--checkpointing_steps epoch  --name pretrainWithC45e-4 --weight_decay 1e-5 --learning_rate   5e-4 --max_train_steps 500000 \
--with_tracking --report_to wandb --streaming True
```
