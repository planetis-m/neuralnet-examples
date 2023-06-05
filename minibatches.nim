# Copyright (c) 2020 Antonis Geralis
import parsecsv, csvutils, strutils, random, math, manu/matrix
{.passC: "-march=native -ffast-math".}

type
  Row = object
    input: array[256, float]
    target: array[10, float]

proc readData: (Matrix[float], Matrix[float]) =
  var
    inputs: seq[float]
    targets: seq[float]
    c: CsvParser
  open(c, "semeion.data", ' ')
  while readRow(c):
    let d = c.row.to(Row)
    inputs.add d.input
    targets.add d.target
  close(c)
  result = (matrix(256, inputs), matrix(10, targets))

proc sigmoid(s: float): float {.inline.} =
  result = 1.0 / (1.0 + exp(-s))
makeUniversal(sigmoid)

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

iterator batches[T](X, Y: Matrix[T], len, batchLen: int): (Matrix[T], Matrix[T]) =
  let n = if batchLen != 0: len div batchLen else: 0
   assert batchLen * n == len
  var batches = newSeq[int16](len)
  for i in 0..<len:
    batches[i] = i.int16
  shuffle(batches)
  for k in countup(0, len-1, batchLen):
    let rows = batches[k ..< k + batchLen]
    yield (X[rows, 0..^1], Y[rows, 0..^1])

proc main =
  const
    len = 1593
    nodes = 28
    rate = 0.01
    term = 0.9
    m = 177
    epochs = 2_000
  let
    (X, Y) = readData()
    sample = X[0..0, 0..^1]
  var
    # Layer 1
    W1 = randNMatrix(X.n, nodes, 0.0, sqrt(2 / X.n))
    b1 = zeros64(1, nodes)
    # Layer 2
    W2 = randNMatrix(nodes, Y.n, 0.0, sqrt(2 / nodes))
    b2 = zeros64(1, Y.n)
    # Momentums
    Ms = (zerosLike(W1), zerosLike(b1), zerosLike(W2), zerosLike(b2))
  for i in 1 .. epochs:
    var loss = 0.0
    for (X, Y) in batches(X, Y, len, m):
      # Foward Prop
      let
        # Layer 1
        Z1 = X * W1 + RowVector64(b1)
        A1 = sigmoid(Z1)
        # Layer 2
        Z2 = A1 * W2 + RowVector64(b2)
        A2 = exp(Z2) /. ColVector64(sumRows(exp(Z2))) # softmax
        # Back Prop
        # Layer 2
        dZ2 = A2 - Y
        db2 = sumColumns(dZ2)
        dW2 = A1.transpose * dZ2
        # Layer 1
        dZ1 = (dZ2 * W2.transpose) *. (1.0 - A1) *. A1
        db1 = sumColumns(dZ1)
        dW1 = X.transpose * dZ1
      # Cross Entropy
      loss = -sum(ln(A2) *. Y)
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
    if i mod 250 == 0:
      echo(" Iteration ", i, ":")
      echo("   Loss = ", formatEng(loss))
      echo("   Prediction =\n", predict(W1, b1, W2, b2, sample))

main()
