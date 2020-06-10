#!/usr/bin/env ruby

require './dezero/layers.rb'

class Model < Layer
  def plot(inputs, to_file='model.png')
    y = self.forward(inputs)
    return plot_dot_graph(y, verbose:true, to_file:to_file)
  end
end


class MLP < Model
  def initialize(fc_output_sizes, activation=method(:sigmoid_simple))
    super()
    @activation = activation
    @layers = []

    fc_output_sizes.each_with_index{|out_size, i|
      layer = LinearLayer.new(out_size)
      layer_name = '@l' + i.to_s
      instance_variable_set(layer_name.intern, layer)
      @layers.append(layer)
    }
  end

  def forward(x)
    @layers.first(@layers.size-1).each{|l|
      x = @activation.call(l.call(x))
    }
    return @layers.last().call(x)
  end
end
