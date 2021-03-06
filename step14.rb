#!/usr/bin/env ruby

require 'test/unit'

class Variable
  def initialize(data)
    if data != nil then
      if not data.is_a?(Array) then
        @data = [data]
        # raise TypeError, data.class.to_s + " is not supported."
      else
        @data = data
      end
    end
    @grad = nil
    @creator = nil
  end
  def set_creator(func)
    @creator = func
  end

  def backward()
    if @grad == nil then
      @grad = @data.clone.fill(1.0)
    end
    funcs = [@creator]
    while funcs != [] do
      f = funcs.pop
      gys = f.outputs.map{|x| x.grad}
      gxs = f.backward(*gys)
      if not gxs.is_a?(Array) then
        gxs = [gxs]
      end
      f.inputs.zip(gxs).each{|x, gx|
        if x.grad === nil then
          x.grad = gx
        else
          puts (x.grad + gx).to_s
          x.grad = [(x.grad + gx).sum]
        end
        if x.creator != nil then
          funcs.push(x.creator)
        end
      }
    end
  end

  def cleargrad()
    @grad = nil
  end

  attr_accessor :data, :grad, :creator
end

class Function
  def call(*inputs)
    xs = inputs.map{|x| x.data}
    ys = forward(*xs)
    if ys.is_a?(Array) then
      ys = [ys]
    end
    outputs = ys.map{|y| Variable.new(y) }

    outputs.each {|output|
      output.set_creator(self)
    }
    @inputs = inputs
    @outputs = outputs
    return outputs.size > 1 ? outputs : outputs[0]
  end

  def forward(x)
    raise NotImplementedError
  end
  def backward(x)
    raise NotImplementedError
  end

  attr_accessor :inputs, :outputs
end

class Square < Function
  def _calc(x)
    if x.is_a?(Array) then
      return x.map{|i| _calc(i) }
    else
      return x ** 2
    end
  end
  def forward(x)
    return x.map{|i| _calc(i)}
  end
  def backward(gy)
    x = @inputs[0].data
    gx = x.zip(gy).map{|i0, i1| i0 * i1 * 2.0}
    return gx
  end
end


class Exp < Function
  def _calc(x)
    if x.is_a?(Array) then
      return x.map{|i| _calc(i) }
    else
      return Math.exp(x)
    end
  end
  def forward(x)
    return x.map{|i| _calc(i)}
  end
  def backward(gy)
    x = @inputs.data
    gx = x.zip(gy).map{|i0, i1| Math.exp(i0) * i1}
    return gx
  end
end


def numerical_diff(f, x, eps=1e-4)
  x0 = Variable.new(x.data.map{|i| i - eps})
  x1 = Variable.new(x.data.map{|i| i + eps})
  y0 = f.call(x0)
  y1 = f.call(x1)
  return (y1.data.zip(y0.data).map{|i1, i0| i1 - i0}).map{|i| i / (2 * eps)}
end

def square(x)
  return Square.new().call(x)
end


def exp(x)
  return Exp.new().call(x)
end

class Add < Function
  def forward(x0, x1)
    y = x0[0] + x1[0]
    return [y]
  end
  def backward(gy)
    return [gy, gy]
  end
end

def add(x0, x1)
  return Add.new().call(x0, x1)
end

begin
  x = Variable.new([3.0])
  y = add(x, x)
  y.backward()
  puts(x.grad)
end

begin
  x = Variable.new([3.0])
  y = add(add(x, x),x)
  y.backward()
  puts(x.grad)
end

begin
  x = Variable.new([3.0])
  y = add(x, x)
  y.backward()
  puts(x.grad)

  x = Variable.new([3.0])
  y = add(add(x, x),x)
  y.backward()
  puts(x.grad)
end
