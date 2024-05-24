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

proc score(predictions, trueLabels: seq[int32]): tuple[accuracy, precision, recall, f1: float] =
  var tp, fp, tn, fn: array[SemeionLabels, int32]
  assert predictions.len == trueLabels.len

  for i in 0..<predictions.len:
    let trueLabel = trueLabels[i]
    let predLabel = predictions[i]

    if trueLabel == predLabel:
      inc tp[trueLabel]
    else:
      inc fp[predLabel]
      inc fn[trueLabel]
    for j in 0..<SemeionLabels:
      if j != trueLabel and j != predLabel:
        inc tn[j]

  var accuracy: float = 0
  var precision, recall, f1: array[SemeionLabels, float]

  for i in 0..<SemeionLabels:
    let tpVal = tp[i].float
    let fpVal = fp[i].float
    let fnVal = fn[i].float
    let tnVal = tn[i].float

    accuracy += (tpVal + tnVal) / (tpVal + fpVal + fnVal + tnVal)

    if tpVal + fpVal > 0:
      precision[i] = tpVal / (tpVal + fpVal)
    else:
      precision[i] = 0

    if tpVal + fnVal > 0:
      recall[i] = tpVal / (tpVal + fnVal)
    else:
      recall[i] = 0

    if precision[i] + recall[i] > 0:
      f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
    else:
      f1[i] = 0

  accuracy = accuracy / SemeionLabels.float
  let avgPrecision = sum(precision) / precision.len.float
  let avgRecall = sum(recall) / recall.len.float
  let avgF1 = sum(f1) / f1.len.float

  result = (accuracy, avgPrecision, avgRecall, avgF1)

type
  KFoldCrossValidation = object
    K: int
    indices: seq[int16]

proc newKFoldCrossValidation(numInstances, numFolds: int): KFoldCrossValidation =
  result = KFoldCrossValidation(K: numFolds)
  result.indices = newSeq[int16](numInstances)
  for i in 0..<numInstances:
    result.indices[i] = i.int16
  shuffle(result.indices)

proc getTrainFold[T](x: KFoldCrossValidation, fold: int, X, Y: Matrix[T]): (Matrix[T], Matrix[T]) =
  let
    dataLen = x.indices.len
    foldLen = if x.K != 0: dataLen div x.K else: 0
    first = fold * foldLen
    last = min(first + foldLen, dataLen)
  var
    trainX = matrixUninit[T](dataLen-foldLen, X.n)
    trainY = matrixUninit[T](dataLen-foldLen, Y.n)
  var c = 0
  for i in 0..<first:
    trainX[[c], 0..^1] = X[[x.indices[i]], 0..^1]
    trainY[[c], 0..^1] = Y[[x.indices[i]], 0..^1]
    inc c
  for i in last..<dataLen:
    trainX[[c], 0..^1] = X[[x.indices[i]], 0..^1]
    trainY[[c], 0..^1] = Y[[x.indices[i]], 0..^1]
    inc c
  result = (trainX, trainY)

proc getTestFold[T](x: KFoldCrossValidation, fold: int, X, Y: Matrix[T]): (Matrix[T], Matrix[T]) =
  let
    dataLen = x.indices.len
    foldLen = if x.K != 0: dataLen div x.K else: 0
    first = fold * foldLen
    last = min(first + foldLen, dataLen)
  var
    testX = matrixUninit[T](foldLen, X.n)
    testY = matrixUninit[T](foldLen, Y.n)
  var c = 0
  for i in first..<last:
    testX[[c], 0..^1] = X[[x.indices[i]], 0..^1]
    testY[[c], 0..^1] = Y[[x.indices[i]], 0..^1]
  result = (testX, testY)

proc main =
  const
    nodes = 28
    rate = 0.01
    beta = 0.9 # decay rate
    epsilon = 1e-8 # avoid division by zero
    m = 177
    epochs = 2_000
    k = 5 # number of folds for cross-validation
  let (X, Y) = readSemeionData()
  # Cross Validation
  let cv = newKFoldCrossValidation(SemeionDataLen, k)
  var metrics: seq[tuple[accuracy, precision, recall, f1: float]] = @[]
  for fold in 0..<k:
    let
      (testX, testY) = cv.getTestFold(fold, X, Y)
      (trainX, trainY) = cv.getTrainFold(fold, X, Y)
    var
      # Layer 1
      W1 = randNMatrix(trainX.n, nodes, 0.0, sqrt(2 / trainX.n))
      b1 = zeros64(1, nodes)
      # Layer 2
      W2 = randNMatrix(nodes, trainY.n, 0.0, sqrt(2 / nodes))
      b2 = zeros64(1, trainY.n)
      # RMSProp
      cache = (zerosLike(W1), zerosLike(b1), zerosLike(W2), zerosLike(b2))
    for i in 1 .. epochs:
      var loss = 0.0
      for (X, Y) in batches(trainX, trainY, trainX.m, m):
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
    # Score
    let predictions = predict(W1, b1, W2, b2, testX)
    let trueLabels = maxIndexRows(testY)
    let foldMetrics = score(predictions, trueLabels)
    metrics.add(foldMetrics)
    echo("Fold ", fold, ": ", foldMetrics)

  # Average
  var avgAccuracy, avgPrecision, avgRecall, avgF1: float32 = 0
  for m in metrics:
    avgAccuracy += m.accuracy
    avgPrecision += m.precision
    avgRecall += m.recall
    avgF1 += m.f1

  avgAccuracy /= k.float32
  avgPrecision /= k.float32
  avgRecall /= k.float32
  avgF1 /= k.float32

  echo("Cross-Validation Results:")
  echo("  Average Accuracy: ", avgAccuracy)
  echo("  Average Precision: ", avgPrecision)
  echo("  Average Recall: ", avgRecall)
  echo("  Average F1 Score: ", avgF1)

main()
