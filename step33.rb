#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy


def f(x)
  y = (x ** 4) - (x ** 2) * 2.0
  return y
end


begin
  x = Variable.new(np.array(2.0))
  y = f(x)
  y.backward()
  puts x.grad.to_s

  gx = x.grad
  x.cleargrad()
  gx.backward()
  puts x.grad.to_s
end


begin
  x = Variable.new(np.array(2.0))
  iters = 10

  for i in 0..iters do
    puts [i, x.to_s].to_s

    y = f(x)
    x.cleargrad()
    y.backward()

    gx = x.grad
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data
  end
end
