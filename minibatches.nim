# Copyright (c) 2020 Antonis Geralis
import std/[parsecsv, strutils, random, math], manu/matrix
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
    Z1 = X * W1 + RowVector[T](b1)
    A1 = sigmoid(Z1)
    # Layer 2
    Z2 = A1 * W2 + RowVector[T](b2)
    A2 = exp(Z2) /. ColVector[T](sumRows(exp(Z2)))
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
    let last = min(k + batchLen, len)
    let rows = batches[k ..< last]
    yield (X[rows, 0..^1], Y[rows, 0..^1])

proc main =
  const
    nodes = 28
    rate = 0.01
    beta = 0.9 # decay rate
    epsilon = 1e-8 # avoid division by zero
    m = 177
    epochs = 2_000
  let
    (X, Y) = readSemeionData()
    sample = X[635..639, 0..^1]
  var
    # Layer 1
    W1 = randNMatrix(X.n, nodes, 0.0, sqrt(2 / X.n))
    b1 = zeros64(1, nodes)
    # Layer 2
    W2 = randNMatrix(nodes, Y.n, 0.0, sqrt(2 / nodes))
    b2 = zeros64(1, Y.n)
    # RMSProp
    cache = (zerosLike(W1), zerosLike(b1), zerosLike(W2), zerosLike(b2))
  for i in 1 .. epochs:
    var loss = 0.0
    for (X, Y) in batches(X, Y, SemeionDataLen, m):
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
      # RMSProp updates
      cache[0] = beta * cache[0] + (1.0 - beta) * (dW1 *. dW1)
      cache[1] = beta * cache[1] + (1.0 - beta) * (db1 *. db1)
      cache[2] = beta * cache[2] + (1.0 - beta) * (dW2 *. dW2)
      cache[3] = beta * cache[3] + (1.0 - beta) * (db2 *. db2)
      # Layer 1
      W1 -= rate * dW1 /. (sqrt(cache[0]) + epsilon)
      b1 -= rate * db1 /. (sqrt(cache[1]) + epsilon)
      # Layer 2
      W2 -= rate * dW2 /. (sqrt(cache[2]) + epsilon)
      b2 -= rate * db2 /. (sqrt(cache[3]) + epsilon)
    # Print progress
    if i mod 250 == 0:
      echo(" Iteration ", i, ":")
      echo("   Loss = ", formatEng(loss))
      echo("   Prediction = ", predict(W1, b1, W2, b2, sample))

main()
