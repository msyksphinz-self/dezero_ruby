#!/usr/bin/env ruby

require './dezero/core_simple.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

def f(x)
  y = x ** 4 - (x ** 2) * 2
  return y
end

def gx2(x)
  return (x ** 2) * 12 -4
end


x = Variable.new(np.array(2.0))
iters = 10

for i in 0..iters do
  puts [i, x.to_s].to_s

  y = f(x)
  x.cleargrad()
  y.backward()

  x.data -= x.grad / gx2(x.data)
end
