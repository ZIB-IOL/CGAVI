# Conditional Gradients for the Approximately Vanishing Ideal

Code for the paper:
[Wirth, E. S., & Pokutta, S. (2022, May). Conditional gradients for the approximately vanishing ideal.
In Proceedings of the International Conference on Artificial Intelligence and Statistics (pp. 2191-2209).
PMLR.](https://proceedings.mlr.press/v151/wirth22a.html)

and

[Wirth, E. and Pokutta, S., 2022. Conditional Gradients for the Approximate Vanishing Ideal.
arXiv preprint arXiv:2202.03349.](https://arxiv.org/abs/2202.03349)


## References
This project is an extension of the previously published release and Git repository
[cgavi](https://github.com/ZIB-IOL/cgavi/releases/tag/v1.0.0) and
[avi_at_scale](https://github.com/ZIB-IOL/avi_at_scale),
respectively.


## Installation guide
Download the repository and store it in your preferred location, say ~/tmp.

Open your terminal and navigate to ~/tmp.

Run the command: 
```shell script
$ conda env create --file environment.yml
```
This will create the conda environment cgavi.

Activate the conda environment with:
```shell script
$ conda activate cgavi
```

Run the tests:
```python3 script
>>> python3 -m unittest
```

No errors should occur.


Execute the experiments: 
```python3 script
>>> python3 experiments_cgavi.py
```

This will create folders named data_frames and plots, which contain subfolders containing the experiment results and 
the plots, respectively. 

The performance experiments can be displayed as latex_code by executing:
```python3 script
>>> experiments_to_latex_cgavi.py
```
