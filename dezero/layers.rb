#!/usr/bin/env ruby

require 'numpy'

class Layer
  def initialize()
    @_params = Set.new()
  end

  def instance_variable_set(name, value)
    if value.is_a?(Parameter) or
      value.is_a?(Layer) then
      @_params.add(name)
    end
    super
  end

  def call(*inputs)
    outputs = self.forward(*inputs)
    if not outputs.is_a?(Array) then
      outputs = [outputs]
    end
    @inputs  = inputs.map{|input| WeakRef.new(input)}
    @outputs = outputs.map{|output| WeakRef.new(output)}
    return outputs.size > 1 ? outputs : outputs[0]
  end

  def forward(x)
    raise NotImplementedError
  end

  def params()
    tmp = @_params.map {|name|
      obj = instance_variable_get(name)
      if obj.is_a?(Layer) then
        obj.params()
      else
        instance_variable_get(name)
      end
    }
    return tmp.flatten
  end

  def cleargrads()
    # puts "  params. name = " + self.params().to_s
    self.params().each{|param|
      param.cleargrad()
    }
  end

  attr_accessor :_params
end


class LinearLayer < Layer
  np = Numpy
  def initialize(out_size, nobias=false, in_size=nil)
    np = Numpy
    super()
    @in_size = in_size
    @out_size = out_size
    @dtype = np.float32

    instance_variable_set(:@w, Parameter.new(nil, 'W'))

    if @in_size != nil then
      self._init_W()
    end

    if nobias then
      instance_variable_set(:@b, nil)
    else
      instance_variable_set(:@b, Parameter.new(np.zeros(out_size, np.float32), 'b'))
    end

  end

  def _init_W()
    np = Numpy
    i = @in_size
    o = @out_size
    w_data = np.random.randn(i, o).astype(@dtype) * np.sqrt(1.0 / i)
    @w.data = w_data
  end

  def forward(x)
    # puts "init1 = " + @w.data.class.to_s
    if @w.data.is_a?(NilClass) then
      @in_size = x.shape[1]
      self._init_W()
    end

    y = linear_simple(x, @w, @b)
    # y = linear(x, @w, @b)
    return y
  end
end
