#!/usr/bin/env ruby

require 'set'
require 'weakref'

require 'pycall'
sys = PyCall.import_module('sys')
p sys.path
require 'numpy'
np = Numpy

$enable_backprop = true


# class Float
#   def *(other)
#     puts "type = " + other.class.to_s
#     if other.is_a?(Variable) then
#       y = other * self
#       return y
#     end
#     return super(other)
#   end
# end

def as_array(x)
  np = Numpy

  if np.isscalar(x)
    return np.array(x)
  end
  return x
end


class Variable
  def initialize(data, name = nil)
    @data = data
    @name = name
    @grad = nil
    @creator = nil
    @generation = 0
  end

  def set_creator(func)
    @creator = func
    @generation = func.generation + 1
  end

  def backward(retain_grad=false, create_graph=true)
    np = Numpy

    if @grad == nil
      @grad = Variable.new(np.ones_like(@data))
    end

    funcs = Array.new()
    seen_set = Set.new

    def add_func(f, funcs, seen_set)
      if f != nil and (not seen_set.include?(f)) then
        funcs.push(f)
        seen_set.add(f)
        funcs.sort!{|a| a.generation}
      end
    end

    add_func(@creator, funcs, seen_set)

    while not funcs.empty? do
      f = funcs.pop
      gys = f.outputs.map{|output| output.grad}

      if create_graph == true then
        gxs = f.backward(*gys)
        if not gxs.is_a?(Array)
          gxs = [gxs]
        end

        f.inputs.zip(gxs).each{|x, gx|
          if x.grad == nil
            x.grad = gx
          else
            x.grad = x.grad + gx
          end
          if x.creator != nil then
            add_func(x.creator, funcs, seen_set)
          end
        }
        if not retain_grad then
          f.outputs.map{|y| y.grad = nil }
        end
      end
    end
  end

  def cleargrad()
    @grad = nil
  end

  def shape()
    return @data.shape
  end

  def ndim()
    return @data.ndim
  end

  def size()
    return @data.size
  end

  def dtype()
    return @data.dtype
  end

  def to_s
    return 'variable(' + @data.to_s + ')'
  end

  def *(other)
    return mul(self, other)
  end

  def /(other)
    return div(self, other)
  end

  def +(other)
    return add(self, other)
  end

  def -(other)
    return sub(self, other)
  end

  def -@
    return neg(self)
  end

  def **(other)
    return pow(self, other)
  end

  attr_accessor :data, :grad, :creator, :generation, :name
end

class Function
  def as_variable(obj)
    if obj.is_a?(Variable) then
      return obj
    end
    return Variable.new(obj)
  end

  def call(*inputs)
    inputs = inputs.map{|x| as_variable(x)}

    xs = inputs.map{|x| x.data}
    ys = forward(*xs)
    if not ys.is_a?(Array) then
      ys = [ys]
    end
    outputs = ys.map{|y| Variable.new(y) }

    if $enable_backprop then
      @generation = (inputs.map{|x| x.generation}).max
      outputs.each{|output|
        output.set_creator(self)
      }
    end

    @inputs = inputs
    @outputs = outputs.map{|output| WeakRef.new(output)}
    return outputs.size > 1 ? outputs : outputs[0]
  end
  def forward(xs)
    raise NotImplementedError
  end
  def backward(gys)
    raise NotImplementedError
  end

  attr_accessor :inputs, :outputs, :generation
end