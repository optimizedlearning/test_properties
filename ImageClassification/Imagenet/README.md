# Installing Packages
1. For BU SCC
Before installing additional packages, we need to set up a virtual environment. Use 'python3 -m venv <env_name>' to create your environment, then 'source <env_name>/bin/activate' to activate.
Then, we need to load some existing modules:
```python3
module load python3/3.8.10 pytorch cuda/11.6
```
To install the rest of the packages, run `pip install -r requirements.txt`. If there are any missing packages, just keep `pip install <package_name>` until there's no error left. 

2. For general purpose
Run `pip install torch` and then `pip install -r requirements.txt`.

# Running the experiments
To run the experiments, use the following command:
```python3
python main.py -a resnet18 --multiprocessing-distributed --world-size 1 --rank 0  /projectnb/aclab/tranhp/Imagenet/ --name imagenetLastEpochIterate
```
We can also change other hyperparameters such as learning rate, momentum, weight decay, or optimizers used, etc,... Refer to main.py for more information.

# Current results
All the results would be stored in results. Currently, we have the plot of multiple properties using 2 consecutive iterates of Imagenet training on Resnet18. For actual data, refer to:
- Consecutive iterates of Imagenet: imagenetConsecRes1/2/3/4 in test_convexity project on wandb.
# Ongoing Experiments:
- Compute the convexity gap, the smoothness constant, and the ratio between every iterate and the last iterate of each epoch.
- Compute the convexity gap, the smoothness constant, and the ratio between every iterate and final iterate.
- Compute large-batch optimal point.
  
