#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

x = Variable.new(np.array(1.0))
y = tanh(x)
x.name = 'x'
y.name = 'y'
y.backward()

iters = 8
for i in 0..(iters-1) do
  gx = x.grad
  x.cleargrad()
  gx.backward()
end

gx = x.grad
puts gx.class
gx.name = 'gx' + iters.to_s
plot_dot_graph(gx, false, 'tanh' + iters.to_s + '.png')
