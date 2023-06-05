# Copyright (c) 2020 Antonis Geralis
import eminim, strutils, math, manu/matrix, streams, parsejson
{.passC: "-march=native -ffast-math".}

type
  HandDigits = object
    input: array[256, float]
    target: array[10, float]

proc readSemeionData: (Matrix64, Matrix64) =
  var
    inputs: seq[float]
    targets: seq[float]
  let fs = newFileStream("semeion.json")
  for x in jsonItems(fs, HandDigits):
    targets.add x.target
    inputs.add x.input
  result = (matrix(256, inputs), matrix(10, targets))

proc sigmoid(s: float): float {.inline.} =
  result = 1.0 / (1.0 + exp(-s))
makeUniversal(sigmoid)

proc relu(s: float): float {.inline.} =
  result = s * float(s > 0)
makeUniversal(relu)

proc loss(y, t: float): float {.inline.} =
  result = t * ln(y) + (1.0 - t) * ln(1.0 - y)
makeUniversalBinary(loss)

proc maxIndexRows[T](m: Matrix[T]): seq[int] =
  result = newSeq[int](m.m)
  for i in 0 ..< m.m:
    var s = 0
    for j in 1 ..< m.n:
      if m[i, j] > m[i, s]: s = j
    result[i] = s

proc predict[T](W1, b1, W2, b2, X: Matrix[T]): seq[int] =
  assert X.m == 1
  let
    # Layer 1
    Z1 = X * W1 + b1
    A1 = sigmoid(Z1)
    # Layer 2
    Z2 = A1 * W2 + b2
    A2 = exp(Z2) / sum(exp(Z2))
  result = maxIndexRows(A2)

template zerosLike[T](a: Matrix[T]): Matrix[T] = matrix[T](a.m, a.n)

proc main =
  const
    nodes = 336
    rate = 0.0006
    term = 0.6
    epochs = 2_000
  let
    (X, Y) = readSemeionData()
    sample = X[0..0, 0..^1]
  var
    # Layer 1
    W1 = randNMatrix(X.n, nodes, 0.0, sqrt(2.0 / float(X.n + nodes)))
    b1 = zeros64(1, nodes)
    # Layer 2
    W2 = randNMatrix(nodes, Y.n, 0.0, sqrt(2.0 / float(nodes + Y.n)))
    b2 = zeros64(1, Y.n)
    # Momentums
    Ms = (zerosLike(W1), zerosLike(b1), zerosLike(W2), zerosLike(b2))
  for i in 1 .. epochs:
    let
      # Foward Prop
      # Layer 1
      Z1 = X * W1 + RowVector64(b1)
      A1 = sigmoid(Z1)
      # Layer 2
      Z2 = A1 * W2 + RowVector64(b2)
      A2 = exp(Z2) /. ColVector64(sumRows(exp(Z2)))
      # Cross Entropy
      loss = -sum(ln(A2) *. Y)
      # Back Prop
      # Layer 2
      dZ2 = A2 - Y
      db2 = sumColumns(dZ2)
      dW2 = A1.transpose * dZ2
      # Layer 1
      dZ1 = (dZ2 * W2.transpose) *. (1.0 - A1) *. A1
      db1 = sumColumns(dZ1)
      dW1 = X.transpose * dZ1
    # Gradient Descent
    # Momentums
    Ms[0] = term * Ms[0] - rate * dW1
    Ms[1] = term * Ms[1] - rate * db1
    Ms[2] = term * Ms[2] - rate * dW2
    Ms[3] = term * Ms[3] - rate * db2
    # Layer 1
    W1 += Ms[0]
    b1 += Ms[1]
    # Layer 2
    W2 += Ms[2]
    b2 += Ms[3]
    # Print progress
    if i mod 500 == 0:
      echo(" Iteration ", i, ":")
      echo("   Loss = ", formatEng(loss))
      echo("   Prediction =\n", predict(W1, b1, W2, b2, sample))

main()
