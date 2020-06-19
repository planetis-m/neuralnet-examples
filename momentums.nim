# Copyright (c) 2019-2020 Antonis Geralis
import math, strutils, ../manu/manu/matrix
{.passC: "-march=native -ffast-math".}

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
      # LAYER 1
      Z1 = X * W1 + RowVector64(b1)
      A1 = sigmoid(Z1)
      # LAYER 2
      Z2 = A1 * W2 + b2
      A2 = sigmoid(Z2)
   result = round(A2)

template zerosLike[T](a: Matrix[T]): Matrix[T] = matrix[T](a.m, a.n)

proc main =
   const
      m = 4 # batch length
      nodes = 5
      rate = 0.5
      term = 0.9
      epochs = 1_000
   let
      X = matrix(2, @[0.0, 0, 0, 1, 1, 0, 1, 1])
      Y = matrix(1, @[0.0, 1, 1, 0])
      # Xavier initialization
      xrange1 = sqrt(6.0 / float(X.n + nodes))
      xrange2 = sqrt(6.0 / float(nodes + Y.n))
   var
      # LAYER 1
      W1 = randMatrix(X.n, nodes, -xrange1..xrange1)
      b1 = zeros64(1, nodes)
      # LAYER 2
      W2 = randMatrix(nodes, Y.n, -xrange2..xrange2)
      b2 = 0.0
      # MOMENTUMS
      Ms = (zerosLike(W1), zerosLike(b1), zerosLike(W2), 0.0)
   for i in 1 .. epochs:
      let
         # Foward Prop
         # LAYER 1
         Z1 = X * W1 + RowVector64(b1)
         A1 = sigmoid(Z1)
         # LAYER 2
         Z2 = A1 * W2 + b2
         A2 = sigmoid(Z2)
         # Cross Entropy
         loss = -sum(loss(A2, Y)) / m.float
         # Back Prop
         # LAYER 2
         dZ2 = A2 - Y
         db2 = sum(dZ2)
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
      # Print progress
      if i mod 250 == 0:
         echo(" Iteration ", i, ":")
         echo("   Loss = ", formatEng(loss))
         echo("   Predictions =\n", predict(W1, b1, W2, b2, X))

main()
