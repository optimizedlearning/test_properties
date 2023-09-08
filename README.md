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
