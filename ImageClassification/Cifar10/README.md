# Running the experiments
To run the experiments, use the following command:
```python3
python main.py --model resnet18 --e 200
```
We can also change other hyperparameters such as learning rate, momentum, weight decay, or optimizers used, etc,... Refer to main.py for more information.
# Current results
- Consecutive: Compute properties of consecutive iterates (consecResnet on wandb).
- final_iterate_cifar:  Compute properties of iterate with respecte to the final training iterate. (finaliterateCifar on wandb).
- last_iterate_epoch: Compute properties of iterate with respecte to the final iterate of each epoch (comparedToEpochIterate on wandb)
# To-do:
- Get large batch iterates.
- Run experiment with fix random seed?.
- Get some bad results.
