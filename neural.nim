# Copyright (c) 2019-2020 Antonis Geralis
import math, strutils, manu/matrix

proc sigmoid(s: float): float {.inline.} =
  result = 1.0 / (1.0 + exp(-s))
makeUniversal(sigmoid)

proc loss(y, t: float): float {.inline.} =
  result = t * ln(y) + (1.0 - t) * ln(1.0 - y)
makeUniversalBinary(loss)

makeUniversal(round)
proc predict[T](W1, b1, W2: Matrix[T], b2: T, X: Matrix[T]): Matrix[T] =
  let
    # Foward Prop
    # Layer 1
    Z1 = X * W1 + RowVector64(b1)
    A1 = sigmoid(Z1)
    # Layer 2
    Z2 = A1 * W2 + b2
    A2 = sigmoid(Z2)
  result = round(A2)

proc main =
  const
    m = 4 # batch length
    nodes = 3
    rate = 0.01
    epochs = 1_000
  let
    X = matrix(2, @[0.0, 0, 0, 1, 1, 0, 1, 1])
    Y = matrix(1, @[0.0, 1, 1, 0])
    # Xavier uniform initialization
    limit1 = sqrt(6.0 / float(X.n + nodes))
    limit2 = sqrt(6.0 / float(nodes + Y.n))
  var
    # Layer 1
    W1 = randMatrix(X.n, nodes, -limit1..limit1)
    b1 = zeros64(1, nodes)
    # Layer 2
    W2 = randMatrix(nodes, Y.n, -limit2..limit2)
    b2 = 0.0
  for i in 1 .. epochs:
    let
      # Foward Prop
      # Layer 1
      Z1 = X * W1 + RowVector64(b1) # broadcast bias to (m, nodes)
      A1 = sigmoid(Z1)
      # Layer 2
      Z2 = A1 * W2 + b2 # scalar to (m, 1)
      A2 = sigmoid(Z2)
      # Cross Entropy
      loss = -sum(loss(A2, Y)) / m.float
      # Back Prop
      # Layer 2
      dZ2 = A2 - Y
      db2 = sum(dZ2)
      dW2 = A1.transpose * dZ2
      # Layer 1
      dZ1 = (dZ2 * W2.transpose) *. (1.0 - A1) *. A1
      db1 = sumColumns(dZ1)
      dW1 = X.transpose * dZ1
    # Gradient Descent
    # Layer 2
    W2 -= rate * dW2
    b2 -= rate * db2
    # Layer 1
    W1 -= rate * dW1
    b1 -= rate * db1
    # Print progress
    if i mod 250 == 0:
      echo(" Iteration ", i, ":")
      echo("   Loss = ", formatEng(loss))
      echo("   Predictions =\n", predict(W1, b1, W2, b2, X))

main()
