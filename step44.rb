#!/usr/bin/env ruby

require './dezero/core.rb'
require './dezero/functions.rb'
require './dezero/util.rb'
require './dezero/layers.rb'

require 'numpy'
np = Numpy

begin
  layer = Layer.new()

  layer.instance_variable_set(:@p1, Parameter.new(np.array(1)))
  layer.instance_variable_set(:@p2, Parameter.new(np.array(2)))
  layer.instance_variable_set(:@p3, Variable.new(np.array(3)))
  layer.instance_variable_set(:@p4, 'test')

  puts layer._params
  puts '--------------'
  layer._params.each{|name|
    puts name, layer.instance_variable_get(name)
  }
end


begin
  def mean_squared_error(x0, x1)
    np = Numpy
    diff = x0 - x1
    return sum(diff ** 2) / as_variable(np.array(diff.size))
  end

  np.random.seed(0)
  x = np.random.rand(100, 1)
  y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
  x = Variable.new(x)
  y = Variable.new(y)

  $l1 = LinearLayer.new(10)
  $l2 = LinearLayer.new(1)

  def predict(x)
    y = $l1.call(x)
    y = sigmoid_simple(y)
    y = $l2.call(y)
    return y
  end

  lr = 0.2
  iters = 10000

  for i in 0..(iters-1) do
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    $l1.cleargrads()
    $l2.cleargrads()
    loss.backward()

    [$l1, $l2].each{|l|
      l.params().each{|p|
        p.data -= lr * p.grad.data
      }
    }

    if i % 1000 == 0 then
      puts loss
    end
  end

end
