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
  
