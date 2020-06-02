#!/usr/bin/env ruby

require './dezero/core_simple.rb'
require './dezero/util.rb'

require 'numpy'
np = Numpy

# class Float
#   def +(other)
#     if other.is_a?(Variable) then
#       return other + self
#     else
#       return self + other
#     end
#   end
# end


# class Integer
#   def +(other)
#     if other.is_a?(Variable) then
#       return other + self
#     else
#       return self + other
#     end
#   end
# end

class Sin < Function
  def forward(x)
    np = Numpy
    y = np.sin(x)
    return y
  end
  def backward(gy)
    np = Numpy
    x = @inputs[0].data
    gx = gy * np.cos(x)
    return gx
  end
end

def sin(x)
  return Sin.new().call(x)
end

begin
  puts("=== Test of Sin Function ===")
  x = Variable.new(np.array(np.pi/4))
  y = sin(x)
  y.backward()

  puts(y.data)
  puts(x.grad)
  puts("=== End Test of Sin Function ===")
end


def factorial(number)
  (1..number).inject(1,:*)
end

def my_sin(x, threshold=0.0001)
  y = Variable.new(0.0)
  for i in (0..100000) do
    c = ((-1) ** i) / factorial(2 * i + 1).to_f
    t = (x ** (2 * i + 1)) * c
    y = y + t
    if t.data.abs < threshold then
      break
    end
  end
  return y
end

begin
  puts("=== Test of MySin Function ===")
  x = Variable.new(np.array(np.pi/4))
  y = my_sin(x, 1e-150)
  y.backward()

  puts(y.data)
  puts(x.grad)

  plot_dot_graph(y, verbose=false, to_file='sin.png')
  puts("=== End Test of My Sin Function ===")
end
