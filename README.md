# MNIST Dataset based Neural Network
This point of this project is to implement the theories I've learned regarding Neural Network. This project is inspired by my own curiosity regarding Neural Network and [Samson Zhang's video creating Neural Network from Scratch](https://www.youtube.com/watch?v=w8yWXqWQYmU&t=1114s). I've learned so much regarding Neural Network from this project. 

## Table of Contents
- [Licence](#licence)
- [Versions](#versions)
- [Installation](#installation)
- [Usage](#usage)
- [Explanation](#explanation)

## Licence
Completely free to use and modify

## Versions
Original Version | V0
- Best Accuracy reached is 0.94 on the 500th training
- No accuracy improver add-ons

## Installation

Clone this repository by using this command on your Terminal

```bash
git clone https://github.com/VinV7/Neural-Network-based-on-MNIST-Dataset.git
```

Afterwards to install the required packages, you first need to create a Virtual Environment inside the cloned folder with this command inside your terminal

Windows
```bash
python -m venv YOUR_VENV_NAME
YOUR_VENV_NAME\Scripts\activate
```

Linux & MacOS
```bash
python3 -m venv YOUR_VENV_NAME
source YOUR_VENV_NAME/bin/activate
```

After that please install the [Setuptools Library](https://pypi.org/project/setuptools/)

Windows
```bash
pip install setuptools
```

Linux & MacOS
```bash
pip3 install setuptools
```

After creating a Virtual Environment and installing [Setuptools Library](https://pypi.org/project/setuptools/), proceed to the requirement packages installation using the setup.py by inputing this command inside the terminal.

```bash
python setup.py install # Use Python3 if your system doesn't recognise regular Python
```

## Usage
In order to run the Neural Network, just press the "Run All" button inside of the nn.ipynb

## Explanation
There are 3 steps of computation in this Neural Network which are Forward Propagation, Backward Propagation and Parameter Updates
### Forward Propagation
In this Forward Propagation, we have 1 input layer which has 784 neurons. One neuron is dedicated for inputing 1 pixel from the (28x28) pixels picture into the First Hidden Layer, so basically we'll need to have the correct matrix size in order for the computation to run correctly therefore we now have the dataset equivalent to the matrix size of (784 x 42000). After that we have 2 hidden layer where the main computation for prediction begin and the matrix size of (784 x 42000) will be changed to (128 x 42000), we have weights and biases which have their own matrix size and will get adjusted later on the Backwards Propagation to get better results. Basically in these hidden layers there are [Matrix Multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) happening. After the computation in the First Hidden Layer are done, it will move to A1 which where the [ReLU function](https://www.dremio.com/wiki/relu-activation-function/) is applied to the calculation to cause non-linearity. Afterwards we'll move to the second hidden layer decreasing the matrix size from (128 x 42000) to (64 x 42000) and the same step will be repeated until our last layer which is the 3rd layer or the output layer which has the size of (10 x 42000). On the output layer we have the matrix size of (10 x 42000) because there are only 10 numbers in this data. Now on those matrixes, the 10 matrixes is basically the 10 output neurons. The highest value on those neurons then will send a signal which will send the output based on their respective neurons, for example neuron 1 has the highest value then the neural network will send the number 0 as the output, since the numbers on the dataset is only from 0 to 9 which is equivalent to 10 numbers, so neuron 1 is equivalent of 0 and so on. I'm sorry if this explanation is bad since this is my first time explaining things this complex.

$$
Z^{[1]} = W^{[1]}X + b^{[1]}
$$

$$
A^{[1]} = g_{\text{ReLU}}(Z^{[1]})
$$

$$
Z^{[2]} = W^{[2]}A^{[1]} + b^{[2]}
$$

$$
A^{[2]} = g_{\text{ReLU}}(Z^{[2]})
$$

$$
Z^{[3]} = W^{[3]}A^{[2]} + b^{[3]}
$$

$$
A^{[3]} = g_{\text{softmax}}(Z^{[3]})
$$

### Backward Propagation
Okay, on this one, Backward propagation is the same as forward propagation but it's backward as the name says. The purpose of backward propagation is to adjust and find the best value to adjust the Weights and Biases so that the output will be more accurate. 

$$
dZ^{[3]} = A^{[3]} - Y
$$

$$
dW^{[3]} = \frac{1}{m} dZ^{[3]} A^{[2]T}
$$

$$
dB^{[3]} = \frac{1}{m} \sum dZ^{[3]}
$$

$$
dZ^{[2]} = W^{[3]T} dZ^{[3]} * g_{\text{ReLU}}' (a^{[2]})
$$

$$
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}
$$

$$
dB^{[2]} = \frac{1}{m} \sum dZ^{[2]}
$$

$$
dZ^{[1]} = W^{[2]T} dZ^{[2]} * g_{\text{ReLU}}' (a^{[1]})
$$

$$
dW^{[1]} = \frac{1}{m} dZ^{[1]} X^{T}
$$

$$
dB^{[1]} = \frac{1}{m} \sum dZ^{[1]}
$$

### Parameter Updates
On parameter updates, as the name says it updates the parameters which are the Weights and Biases. On this one it's really simple. The formula shows us that the biases and the weights minus the alpha and the backward version of the weights and biases. The alpha value is the learning rates value which you can adjust when calling the Neural Network class. Oh yeah, the value and the biases are randomised matrix values based on their respective matrix size. 

$$
W^{[3]} = W^{[3]} - \alpha dW^{[3]}
$$

$$
b^{[3]} = b^{[3]} - \alpha db^{[3]}
$$

$$
W^{[2]} = W^{[2]} - \alpha dW^{[2]}
$$

$$
b^{[2]} = b^{[2]} - \alpha db^{[2]}
$$

$$
W^{[1]} = W^{[1]} - \alpha dW^{[1]}
$$

$$
b^{[1]} = b^{[1]} - \alpha db^{[1]}
$$

## Contributing

Pull requests are welcome. For major changes and improvements

## References 

### External Libraries

- [Pandas](https://pandas.pydata.org/): Used for data manipulation and analysis.
- [NumPy](https://numpy.org/): Fundamental package for scientific computing with Python.
- [Matplotlib](https://matplotlib.org/): Plotting library for creating visualizations in Python.

### Online Resources

- [ChatGPT](https://chatgpt.com): Helping as a guide in this project.
- [StackOverFlow](https://stackoverflow.com): As a guide in solving problems in this project.

