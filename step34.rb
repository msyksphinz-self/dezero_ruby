#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

x = Variable.new(np.array(1.0))
y = sin(x)
y.backward()

for i in 0..2 do
  gx = x.grad
  x.cleargrad()
  gx.backward()
  puts x.grad.to_s
end

plt = PyCall.import_module('matplotlib.pyplot')

x = Variable.new(np.linspace(-7, 7, 200))
y = sin(x)
y.backward()

logs = [y.data.flatten()]

for i in 0..2 do
  logs.push(x.grad.data.flatten())
  gx = x.grad
  x.cleargrad()
  gx.backward()
end

labels = ["y=sin(x)", "y'", "y'", "y'''"]

logs.zip(labels).each{|log, l|
  plt.plot(x.data, log, label:l)
}
plt.legend(loc='lower right')
plt.show()
