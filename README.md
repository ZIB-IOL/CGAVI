# Conditional Gradients For The Approximately Vanishing Ideal 

## Installation guide
Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command: 
```shell script
$ conda env create --file environment.yml
```
This will create the conda environment approximately_vanishing_ideal.

Activate the conda environment with:
```shell script
$ conda activate approximately_vanishing_ideal
```

In the file global_.py, change the value of gpu_memory_ to the maximum amount of gpu you wish to use to perform
computations.

Run the tests:
```python3 script
>>> python3 -m unittest
```

No errors should occur.

Execute the experiments: 
```python3 script
>>> python3 experiments.py
```

This will create a folder named data_frames, which continues subfolders containing the experiment results. 

The experiments can be displayed as latex_code by executing:
```python3 script
>>> experiments_results.py
```
