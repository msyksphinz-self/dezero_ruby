#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'
require './dezero/layers.rb'
require './dezero/models.rb'

require 'numpy'
np = Numpy

class TwoLayerNet < Model
  def initialize(hidden_size, out_size)
    super()
    instance_variable_set(:@l1, LinearLayer.new(hidden_size))
    instance_variable_set(:@l2, LinearLayer.new(out_size))
  end

  def forward(x)
    y = sigmoid_simple(@l1.call(x))
    y = @l2.call(y)
    return y
  end
end

x = Variable.new(np.random.randn(5, 10), name='x')
model = TwoLayerNet.new(100, 10)
model.plot(x)


begin
  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
  x = Variable.new(x)
  y = Variable.new(y)

  lr = 0.2
  max_iter = 10000
  hidden_size = 10


  model = TwoLayerNet.new(hidden_size, 1)

  for i in 0..(max_iter-1) do
    y_pred = model.call(x)
    loss = mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    model.params().each{|p|
      p.data -= lr * p.grad.data
    }

    if i % 1000 == 0 then
      puts loss
    end
  end
end


begin
  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
  x = Variable.new(x)
  y = Variable.new(y)

  lr = 0.2
  max_iter = 10000
  hidden_size = 10

  model = MLP.new([10, 1])

  for i in 0..(max_iter-1) do
    y_pred = model.call(x)
    loss = mean_squared_error(y, y_pred)

    model.cleargrads()
    loss.backward()

    model.params().each{|p|
      p.data -= lr * p.grad.data
    }

    if i % 1000 == 0 then
      puts loss
    end
  end
end
