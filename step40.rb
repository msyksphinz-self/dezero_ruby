#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

begin
  x0 = Variable.new(np.array([1, 2, 3]))
  x1 = Variable.new(np.array([10]))
  y = x0 + x1
  puts y

  y.backward
  puts x1.grad
end
