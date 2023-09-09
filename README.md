# Project overview
In this project, we would like to check some common properties of the loss functions during training deep learning tasks. In theory, we usually assume the loss to be convex and smooth but that might not be the case due to deep neural networks. For simplicity, let us denote our model parameter as $x,y \in R^d$ and our loss function $f(x): R^d \mapsto R$. Then we want to check the following.
+ Convexity_gap: We compute the additive convexity gap in every iterate as $f(x_t) -f(y) - \langle \nabla f(x_t), x_t-y  \rangle$ where $x_t$ is the current iterate and $y$ is some reference point. We then report the average of this quantity in every epoch (negative convexity gap means the function is convex).
+ Smoothness:  We compute the smoothness constant $L= \|x_t -\nabla f(x_t)\|/\| \|y -\nabla f(y)\|$ where $x_t$ is the current iterate and $y$ is some reference point. We then report the maximum L of every epoch.
+ Ratio: We also compute the multiplicative convexity gap which is $\langle \nabla f(x_t), x_t-y  \rangle/(f(x_t) -f(y)) $. We then report the sum of the numerator/sum of the denominator in each epoch (our function is "well-behaved" if this ratio is a positive constant).
# Installing Packages
1. For BU SCC
   
Before installing additional packages, we need to set up a virtual environment. Use 'python3 -m venv <env_name>' to create your environment, then 'source <env_name>/bin/activate' to activate.
Then, we need to load some existing modules:
```python3
module load python3/3.8.10 pytorch cuda/11.6
```
To install the rest of the packages, go to the appropriate project and run `pip install -r requirements.txt`. If there are any missing packages, just keep `pip install <package_name>` until there's no error left. 

2. For general purpose
   
Run `pip install torch` and then `pip install -r requirements.txt`.
# Checkpoints
The current checkpoints for Cifar10 and Imagenet are saved as dictionaries in /projectnb/aclab/tranhp/experiments/cifar10_resnet/checkpoint and /projectnb/aclab/tranhp/Imagenet/checkpoint respectively. All checkpoints have 4 keys: "state_dict", "current_loss", "prev_loss", and "model_dict":
+  "state_dict": Optimizer dictionary in the last iterate of every epoch.
+ "current_loss": The loss evaluated at the last iterate of every epoch.
+ "prev_loss": The loss evaluated at the second to last iterate of every epoch.
+ "model_dict": Model dictionary dictionary in the last iterate of every epoch.
To access the checkpoints, use `torch.load(path_to_checkpoint)`.
# Modifications on optimizers
Compared to regular optimizers, we have a few extra state parameters:
+ "prev_param": To store any param that we would like to keep track of (usually the param at time t before we call optimizer.step() to get to the iterate t+1).
+ "prev_grad": Gradient evaluated at prev_param.
+  "prev_prev_param": Param 1 iterate before prev_param.
+  "prev_prev_grad": Gradient evaluated at prev_prev_param.
We also have a few extra functions:
+ `get_params()`: to access params and grads.
+  `check_convexity()`: To compute the quantity $\langle \nabla f(x_t), x_t-y  \rangle$ where $x_t$ is the current iterate and $y$ is some reference point.
+  `check_smoothness()`: To compute the quantity $\|x_t -\nabla f(x_t)\|/\| \|y -\nabla f(y)\|$ where $x_t$ is the current iterate and $y$ is some reference point.
+  `save_param()`: To save any params that we want in "prev_param" and "prev_grad".
+  `save_prev_param()`: Same as `save_param()` but for "prev_prev_param" and "prev_prev_grad"
