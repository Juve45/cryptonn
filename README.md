# Attack on FE-powered Neural Networks 



## Running the attack on FE-powered NN

1. Generate first layer weights multiplied by the input samples using the command:
`python src/gen_wx.py`

or with any other method using the NN to be attacked and save the results into `data`:
- `a.data` should be the result of `WX`
- `w.data` and `x.data` can be used for analyzing the result.

2. Run `python linprog.py` in order to operate the attack


## Running the samples

- install charm
- download MNIST/CIFAR10 Dataset








