#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  y = reshape(x, [6])
  puts y.to_s
  y.backward()
  puts x.grad.to_s
end


begin
  x = Variable.new(np.random.randn(1, 2, 3))
  puts x.reshape([2, 3])
  puts x.reshape(2, 3)
end


begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  y = transpose(x)
  y.backward()
  puts x.grad.to_s
end


begin
  x = Variable.new(np.random.rand(2, 3))
  puts x.transpose()
  puts x.T()
end
