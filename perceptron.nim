# Copyright (c) 2020 Antonis Geralis
import random, strutils, manu/matrix

proc label(s: float): float {.inline.} =
   result = if s == 0: -1 else: 1
makeUniversal(label)

proc heaviside(s: float): float {.inline.} =
   result = if s >= 0: 1 else: -1
makeUniversal(heaviside)

proc hinge(y, t: float): float {.inline.} =
   result = max(0, 1 - t * y)
makeUniversalBinary(hinge)

proc predict(s: float): float {.inline.} =
   result = if s >= 0: 1 else: 0
makeUniversal(predict)

proc predict[T](W: Matrix[T], b: T, X: Matrix[T]): Matrix[T] =
   let
      Z = X * W + b
      A = predict(Z)
   result = A

proc main =
   const
      m = 4
      epochs = 30
      rate = 0.01
   let
      X = matrix(2, @[0.0, 0, 0, 1, 1, 0, 1, 1])
      Y = matrix(1, @[0.0, 1, 1, 1]).label
   var
      W = randMatrix(2, 1, -1.0..1.0)
      b = 0.0
   for i in 1 .. epochs:
      # Foward Prop
      let
         Z = X * W + b
         A = heaviside(Z)
      # Cross Entropy
      let loss = sum(hinge(A, Y)) / m.float
      # Back Prop
      let
         dZ = A - Y
         db = sum(dZ)
         dW = X.transpose * dZ
      # Gradient Descent
      W -= rate * dW
      b -= rate * db
      if i mod 5 == 0:
         echo(" Iteration ", i, ":")
         echo("   Loss = ", formatEng(loss))
         echo("   Predictions =\n", predict(W, b, X))

main()
