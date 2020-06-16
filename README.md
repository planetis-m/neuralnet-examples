# Neural net examples
In this repo  you will find examples of neural networks implimented using
matrices. This means, they're vectorised versions of the algorithms built
from scratch, using the tutorial linked at the bottom. An expirement that
helped identify missing futures, bugs and usability problems in my matrix
[library](https://github.com/b3liever/manu).

## Examples included
The examples follow a progression of sophistication, layer count and
function. Done for comparison.

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
activation function. XOR gate. todo: "He normal initialization" for weights.

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
   "layers": [2, 3, 1],
   "activation_function": ["sigmoid", "sigmoid"]
}
```

### 4. [exseimion](exseimion.nim)
Handwritten digit classification is a multi-label classification problem.
The data set used is [semeion.data](http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data).

```json
{
   "layers": [256, 336, 10],
   "activation_function": ["sigmoid", "softmax"]
}
```

### 5. [minibatches](minibatches.nim)
Same as previous, Data is split in small batches (subsets). Impoves memory
efficiency, accuracy for a trade-off in compute efficiency. todo: avoid copying?

```json
{
   "layers": [256, 336, 10],
   "activation_function": ["sigmoid", "softmax"]
}
```

## Acknowledgments
- [How to implement a neural network](https://peterroelants.github.io/posts/neural-network-implementation-part01/)
- [Machine Learning FAQ](https://sebastianraschka.com/faq/index.html)

## License
This library is distributed under the [MIT license](LICENSE).
