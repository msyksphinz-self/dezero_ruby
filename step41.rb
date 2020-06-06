#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

begin
  x = Variable.new(np.random.randn(2, 3))
  w = Variable.new(np.random.randn(3, 4))
  y = matmul(x, w)
  y.backward()

  puts x.grad.shape.to_s
  puts w.grad.shape.to_s
end
