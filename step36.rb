#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

x = Variable.new(np.array(2.0))
y = x ** 2
y.backward()
gx = x.grad
x.cleargrad()

z = gx ** 3 + y
z.backward()
puts x.grad
