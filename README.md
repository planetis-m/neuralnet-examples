# Neural net examples

In this repo  you will find examples of neural networks implimented from scratch using
my matrix [library](https://github.com/b3liever/manu).

## Examples included

### 1. [perceptron](perceptron.nim)

The [perceptron](https://en.wikipedia.org/wiki/Perceptron) algorithm.
The activation function is a binary step function called "heaviside step
function". Hinge loss as a loss function. Capable of binary
classification. The example functions as an OR gate.

```json
{
   "layers": [2, 1],
   "activation_function": ["heaviside"]
}
```

### 2. [neural](neural.nim)

Two layer neural network with gradient descent. Both layers use sigmoid as
activation function. XOR gate. Weights are initialized from a uniform
distribution ``U(-sqrt(6 / (in + out)), sqrt(6 / (in + out)))`` (Xavier initialization).

```json
{
   "layers": [2, 3, 1],
   "activation_function": ["sigmoid", "sigmoid"]
}
```

### 3. [momentums](momentums.nim)

Same as previous except implemented the momentum method. It impovers training
speed and accuracy (avoid getting stuck in a local minima). XOR gate

```json
{
   "layers": [2, 5, 1],
   "activation_function": ["sigmoid", "sigmoid"]
}
```

### 4. [exseimion](exseimion.nim)

Handwritten digit classification is a multi-label classification problem.
The data set used is [semeion.data](http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data).

```json
{
   "layers": [256, 51, 10],
   "activation_function": ["sigmoid", "softmax"]
}
```

### 5. [minibatches](minibatches.nim)

Same as the previous, data is split in small batches (subsets). Impoves memory
efficiency, accuracy for a trade-off in compute efficiency. Uses root mean squared propagation,
instead of SGD.

```json
{
   "layers": [256, 51, 10],
   "activation_function": ["sigmoid", "softmax"]
}
```

### Bonus: [Cross Validation](cross_validation.nim)

Same as the previous, but with cross validation. Implements accuracy,
precision, recall and f1-score metrics.

```json
{
   "layers": [256, 51, 10],
   "activation_function": ["sigmoid", "softmax"]
}
```

**DISCLAIMER**: Only for learning purposes. Nim has its own machine learning
framework [Arraymancer](https://github.com/mratsim/Arraymancer) as well as
[torch bindings](https://github.com/SciNim/flambeau).

## Acknowledgments

- [UFLDL Tutorial](http://ufldl.stanford.edu/tutorial/)
- [How to implement a neural network](https://peterroelants.github.io/posts/neural-network-implementation-part01/)
- [Machine Learning FAQ](https://sebastianraschka.com/faq/index.html)

## License

This library is distributed under the [MIT license](LICENSE).
