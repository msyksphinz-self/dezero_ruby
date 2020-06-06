#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

np.random.seed(0)
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)
x = Variable.new(x)
y = Variable.new(y)

$w = Variable.new(np.zeros([1, 1]))
$b = Variable.new(np.zeros(1))

def predict(x)
  y = matmul(x, $w) + $b
  return y
end

def mean_squared_error(x0, x1)
  diff = x0 - x1
  return sum(diff ** 2) / diff.size
end

lr = 0.1
iters = 100

for i in 0..(iters-1) do
  y_pred = predict(x)
  loss = mean_squared_error(y, y_pred)

  $w.cleargrad()
  $b.cleargrad()
  loss.backward()

  $w.data -= lr * $w.grad.data
  $b.data -= lr * $b.grad.data

  puts $w, $b, loss
end
