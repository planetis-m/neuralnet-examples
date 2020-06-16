# Copyright (c) 2020 Antonis Geralis
import parsecsv, csvutils, strutils, math, manu/matrix
{.passC: "-march=native -ffast-math".}

type
   Row = object
      input: array[256, float]
      target: array[10, float]

proc readData: (Matrix64, Matrix64) =
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

proc loss(y, t: float): float {.inline.} =
   result = t * ln(y) + (1.0 - t) * ln(1.0 - y)
makeUniversalBinary(loss)

proc predict[T](W1, b1, W2, b2, X: Matrix[T]): Matrix[T] =
   assert X.m == 1
   let
      # LAYER 1
      Z1 = X * W1 + b1
      A1 = sigmoid(Z1)
      # LAYER 2
      Z2 = A1 * W2 + b2
      A2 = exp(Z2) / sum(exp(Z2))
   result = A2

template zerosLike[T](a: Matrix[T]): Matrix[T] = matrix[T](a.m, a.n)

proc main =
   const
      nodes = 336
      rate = 0.0006
      term = 0.6
      epochs = 2_000
   let
      (X, Y) = readData()
      sample = X[0..0, 0..^1]
   var
      # LAYER 1
      W1 = randMatrix(256, nodes, -1.0..1.0)
      b1 = zeros64(1, nodes)
      # LAYER 2
      W2 = randMatrix(nodes, 10, -1.0..1.0)
      b2 = zeros64(1, 10)
      # MOMENTUMS
      Ms = (zerosLike(W1), zerosLike(b1), zerosLike(W2), zerosLike(b2))
   for i in 1 .. epochs:
      # Foward Prop
      let
         # LAYER 1
         Z1 = X * W1 + RowVector64(b1)
         A1 = sigmoid(Z1)
         # LAYER 2
         Z2 = A1 * W2 + RowVector64(b2)
         A2 = exp(Z2) /. ColVector64(sumRows(exp(Z2)))
      # Back Prop
      let
         # LAYER 2
         dZ2 = A2 - Y
         db2 = sumColumns(dZ2)
         dW2 = A1.transpose * dZ2
         # LAYER 1
         dZ1 = (dZ2 * W2.transpose) *. (1.0 - A1) *. A1
         db1 = sumColumns(dZ1)
         dW1 = X.transpose * dZ1
      # Gradient Descent
      # MOMENTUMS
      Ms[0] = term * Ms[0] - rate * dW1
      Ms[1] = term * Ms[1] - rate * db1
      Ms[2] = term * Ms[2] - rate * dW2
      Ms[3] = term * Ms[3] - rate * db2
      # LAYER 1
      W1 += Ms[0]
      b1 += Ms[1]
      # LAYER 2
      W2 += Ms[2]
      b2 += Ms[3]
      # Cross Entropy
      let loss = -sum(ln(A2) *. Y)
      if i mod 500 == 0:
         echo(" Iteration ", i, ":")
         echo("   Loss = ", formatEng(loss))
         echo("   Prediction =\n", predict(W1, b1, W2, b2, sample))

main()
