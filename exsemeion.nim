# Copyright (c) 2020 Antonis Geralis
import std/[parsecsv, strutils, math], manu/matrix
{.passC: "-march=native -ffast-math".}

const
  SemeionDataLen = 1593
  SemeionAttributes = 256
  SemeionLabels = 10

proc readSemeionData: (Matrix[float], Matrix[float]) =
  var p: CsvParser
  try:
    open(p, "semeion.data", ' ')
    var
      inputs = newSeq[float](SemeionAttributes*SemeionDataLen)
      targets = newSeq[float](SemeionLabels*SemeionDataLen)
    var x = 0
    while readRow(p):
      for y in 0..<SemeionAttributes:
        inputs[x * SemeionAttributes + y] = parseFloat(p.row[y])
      for y in 0..<SemeionLabels:
        targets[x * SemeionLabels + y] = parseFloat(p.row[SemeionAttributes + y])
      inc x
    result = (matrix(SemeionAttributes, inputs), matrix(SemeionLabels, targets))
  finally:
    close(p)

proc sigmoid(s: float): float {.inline.} =
  result = 1.0 / (1.0 + exp(-s))
makeUniversal(sigmoid)

proc relu(s: float): float {.inline.} =
  result = s * float(s > 0)
makeUniversal(relu)

proc loss(y, t: float): float {.inline.} =
  result = t * ln(y) + (1.0 - t) * ln(1.0 - y)
makeUniversalBinary(loss)

proc maxIndexRows[T](m: Matrix[T]): seq[int32] =
  result = newSeq[int32](m.m)
  for i in 0 ..< m.m:
    var s: int32 = 0
    for j in 1 ..< m.n:
      if m[i, j] > m[i, s]: s = j.int32
    result[i] = s

proc predict[T](W1, b1, W2, b2, X: Matrix[T]): seq[int32] =
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
    nodes = 51 # heuristic: square root of the product of the input and output sizes
    rate = 0.0006
    term = 0.6
    epochs = 2_000
  let
    (X, Y) = readSemeionData()
    sample = X[635..639, 0..^1]
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
      echo("   Prediction = ", predict(W1, b1, W2, b2, sample))

main()
