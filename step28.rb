#!/usr/bin/env ruby

require './dezero/core_simple.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

def rosenbrock(x0, x1)
  y = ((x1 - x0 ** 2) ** 2) * 100 + (x0 - 1) ** 2
  return y
end

begin
  x0 = Variable.new(np.array(0.0))
  x1 = Variable.new(np.array(2.0))

  y = rosenbrock(x0, x1)
  y.backward()
  puts [x0.grad, x1.grad].to_s
end

begin
  x0 = Variable.new(np.array(0.0))
  x1 = Variable.new(np.array(2.0))
  lr = 0.001
  iters = 50000
  for i in 0..iters do
    puts [x0.to_s, x1.to_s].to_s

    y = rosenbrock(x0, x1)
    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
  end
end
