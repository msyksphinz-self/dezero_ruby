#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  y = sin(x)
  puts (y)
end


begin
  x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
  c = Variable.new(np.array([[10, 20, 30], [40, 50, 60]]))
  y = x + c
  puts (y)
end

# begin
#   x = Variable.new(np.array([[1, 2, 3], [4, 5, 6]]))
#   c = Variable.new(np.array([[10, 20, 30], [40, 50, 60]]))
#   t = x + c
#   y = sum(t)
#   y.backward()
#   puts y.grad
#   puts t.grad
#   puts x.grad
#   puts c.grad
# end
