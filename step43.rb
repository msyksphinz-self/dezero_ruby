#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x = Variable.new(x)
y = Variable.new(y)

i = 1
h = 10
o = 1
$w1 = Variable.new(0.01 * np.random.randn(i, h))
$b1 = Variable.new(np.zeros(h))
$w2 = Variable.new(0.01 * np.random.randn(h, o))
$b2 = Variable.new(np.zeros(o))

def predict(x)
  y = linear_simple(x, $w1, $b1)
  y = sigmoid_simple(y)
  y = linear_simple(y, $w2, $b2)
  return y
end

def mean_squared_error(x0, x1)
  np = Numpy
  diff = x0 - x1
  return sum(diff ** 2) / as_variable(np.array(diff.size))
end

lr = 0.2
iters = 10000

for i in 0..(iters-1) do
  y_pred = predict(x)
  loss = mean_squared_error(y, y_pred)

  $w1.cleargrad()
  $b1.cleargrad()
  $w2.cleargrad()
  $b2.cleargrad()
  loss.backward()

  $w1.data -= lr * $w1.grad.data
  $b1.data -= lr * $b1.grad.data
  $w2.data -= lr * $w2.grad.data
  $b2.data -= lr * $b2.grad.data

  if i % 1000 == 0 then
    puts loss
  end
end
