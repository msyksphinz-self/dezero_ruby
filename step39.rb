#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

begin
  x = Variable.new(np.array([1, 2, 3, 4, 5, 6]))
  y = sum(x)
  y.backward()
  puts y
  puts x.grad
end

begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  y = sum(x)
  y.backward()
  puts y
  puts x.grad
end


begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  y = sum(x, 0)
  y.backward()
  puts y
  puts x.grad

  x = Variable.new(np.random.randn(2, 3, 4, 5))
  y = x.sum(keepdims: true)
  puts y.shape
end
