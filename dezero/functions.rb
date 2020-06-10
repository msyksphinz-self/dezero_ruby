class Square < Function
  def forward(x)
    np = Numpy
    y = x ** 2
    return np.array([y])
  end
  def backward(gy)
    x = @inputs[0]
    gx = 2 * x * gy
    return gx
  end
end

def square(x)
  return Square.new().call(x)
end


class Exp < Function
  def forward(x)
    # puts "  Exp.shape = " + x.shape.to_s
    np = Numpy
    y = np.exp(x)
    # puts "  Exp.shape = " + y.shape.to_s
    return y
  end
  def backward(gy)
    np = Numpy
    y = @outputs[0].__getobj__
    # puts "backward.Exp y = " + y.class.to_s
    gx = gy * y
    return gx
  end
end

def exp(x)
  return Exp.new().call(x)
end


class Add < Function
  def forward(x0, x1)
    # puts "  Add.shape = " + x0.shape.to_s + ", " + x1.shape.to_s
    np = Numpy
    @x0_shape = x0.shape()
    @x1_shape = x1.shape()
    y = x0 + x1
    # puts "  Add.shape = " + y.shape.to_s
    return y
  end
  def backward(gy)
    gx0 = gy
    gx1 = gy
    if @x0_shape != @x1_shape then
      gx0 = sum_to(gx0, @x0_shape)
      gx1 = sum_to(gx1, @x1_shape)
    end
    return [gx0, gx1]
  end
end

def add(x0, x1)
  x1 = as_array(x1)
  return Add.new().call(x0, x1)
end


class Sub < Function
  def forward(x0, x1)
    np = Numpy
    @x0_shape = x0.shape
    @x1_shape = x1.shape
    y = x0 - x1
    return y
  end
  def backward(gy)
    gx0 = gy
    gx1 = -gy
    if @x0_shape != @x1_shape then
      gx0 = sum_to(gx0, @x0_shape)
      gx1 = sum_to(gx1, @x1_shape)
    end
    return [gx0, gx1]
  end
end

def sub(x0, x1)
  return Sub.new().call(x0, x1)
end

class Mul < Function
  def forward(x0, x1)
    # puts "  shape = " + x0.shape.to_s + ", " + x1.shape.to_s
    np = Numpy
    y = np.array(x0 * x1)
    # puts "  shape = " + y.shape.to_s
    return y
  end
  def backward(gy)
    x0 = @inputs[0]
    x1 = @inputs[1]
    gx0 = gy * x1
    gx1 = gy * x0
    if x0.shape != x1.shape then
      gx0 = sum_to(gx0, x0.shape)
      gx1 = sum_to(gx1, x1.shape)
    end
    return [gx0, gx1]
  end
end

def mul(x0, x1)
  x1 = as_array(x1)
  return Mul.new().call(x0, x1)
end


class Div < Function
  def forward(x0, x1)
    # puts "  Div.shape = " + x0.shape.to_s + ", " + x1.shape.to_s
    np = Numpy
    y = np.array(x0 / x1)
    # puts "  Div.shape = " + y.shape.to_s
    return y
  end
  def backward(gy)
    x0 = @inputs[0]
    x1 = @inputs[1]
    gx0 = gy / x1
    gx1 = gy * (-x0 / x1 ** 2.0)
    if x0.shape != x1.shape then
      gx0 = sum_to(gx0, x0.shape)
      gx1 = sum_to(gx1, x1.shape)
    end
    return [gx0, gx1]
  end
end

def div(x0, x1)
  return Div.new().call(x0, x1)
end


class Neg < Function
  def forward(x)
    # puts "  Neg.shape = " + x.shape.to_s
    np = Numpy
    y = np.array(x * -1.0)
    # puts "  Neg.shape = " + y.shape.to_s
    return y
  end
  def backward(gy)
    return -gy
  end
end

def neg(x)
  return Neg.new().call(x)
end


class Pow < Function
  def initialize(c)
    @c = c
  end
  def forward(x)
    np = Numpy
    y = x ** @c
    return y
  end
  def backward(gy)
    x = @inputs[0]
    c = @c
    gx = (x ** (c - 1) * gy) * c
    return gx
  end
end

def pow(x, c)
  return Pow.new(c).call(x)
end


class Sin < Function
  def forward(x)
    np = Numpy
    y = np.sin(x)
    return np.array([y])
  end
  def backward(gy)
    np = Numpy
    x = @inputs[0]
    gx = gy * cos(x)
    return gx
  end
end

def sin(x)
  return Sin.new().call(x)
end


class Cos < Function
  def forward(x)
    np = Numpy
    y = np.cos(x)
    return np.array([y])
  end
  def backward(gy)
    np = Numpy
    x = @inputs[0]
    gx = gy * sin(x) * -1
    return gx
  end
end

def cos(x)
  return Cos.new().call(x)
end


class Tanh < Function
  def forward(x)
    np = Numpy
    y = np.tanh(x)
    return np.array([y])
  end
  def backward(gy)
    y = @outputs[0].__getobj__
    gx = gy * (y * y * (-1) + 1)
    return gx
  end
end

def tanh(x)
  return Tanh.new().call(x)
end

class Reshape < Function
  def initialize(shape)
    @shape = shape
  end

  def forward(x)
    np = Numpy
    @x_shape = x.shape
    y = x.reshape(@shape)
    return y
  end

  def backward(gy)
    return reshape(gy, @x_shape)
  end
end

def reshape(x, shape)
  if x.shape == shape
    return as_variable(x)
  end
  return Reshape.new(shape).call(x)
end


class Transpose < Function
  def forward(x)
    np = Numpy
    y = np.transpose(x)
    return y
  end

  def backward(gy)
    gx = transpose(gy)
    return gx
  end
end

def transpose(x)
  return Transpose.new().call(x)
end


class Sum < Function
  def initialize(axis, keepdims)
    @axis = axis
    @keepdims = keepdims
  end

  def forward(x)
    # puts "  Sum.shape = " + x.shape.to_s
    np = Numpy
    @x_shape = x.shape
    y = x.sum(axis:@axis, keepdims:@keepdims)
    y = np.array(y)
    # puts "  Sum.shape = " + y.shape.to_s
    return y
  end

  def backward(gy)
    # puts "  shape = " + gy.shape.to_s
    gy = reshape_sum_backward(gy, @x_shape, @axis, @keepdims)
    gx = broadcast_to(gy, @x_shape)
    # puts "  shape = " + gx.shape.to_s
    return gx
  end
end

def sum(x, axis=nil, keepdims=false)
  return Sum.new(axis, keepdims).call(x)
end


class BroadcastTo < Function
  def initialize(shape)
    @shape = shape
  end

  def forward(x)
    np = Numpy
    @x_shape = x.shape
    y = np.broadcast_to(x, @shape)
    return y
  end

  def backward(gy)
    gx = sum_to(gy, @x_shape)
    return gx
  end
end


def broadcast_to(x, shape)
  if x.shape == shape then
    return as_variable(x)
  end
  return BroadcastTo.new(shape).call(x)
end


class SumTo < Function
  def initialize(shape)

    @shape = shape
  end

  def forward(x)
    # puts "  SumTo.shape = " + x.shape.to_s
    np = Numpy
    @x_shape = x.shape
    y = util_sum_to(x, @shape)
    # puts "  SumTo.shape = " + y.shape.to_s
    return y
  end

  def backward(gy)
    gx = broadcast_to(gy, @x_shape)
    return gx
  end
end


def sum_to(x, shape)
  if x.shape == shape then
    return as_variable(x)
  end
  # puts "sum_to : shape = " + shape.to_s
  return SumTo.new(shape).call(x)
end


class MatMul < Function
  def forward(x, w)
    # puts "  MatMul.shape = " + x.shape.to_s + ", " + w.shape.to_s
    y = x.dot(w)
    # puts "  MalMul.shape = " + y.shape.to_s
    return y
  end

  def backward(gy)
    x = @inputs[0]
    w = @inputs[1]
    gx = matmul(gy, w.T)
    gW = matmul(x.T, gy)
    return gx, gW
  end
end

def matmul(x, w)
  return MatMul.new().call(x, w)
end


class Linear < Function
  def forward(x, w, b)
    puts "Linear.Forward"
    y = x.dot(w)
    if b != nil then
      y += b
    end
    return y
  end
  def backward(gy)
    puts "Linear.Backward"
    x = @inputs[0]
    w = @inputs[1]
    b = @inputs[2]
    gb = b.data == nil ? nil : sum_to(gy, b.shape)
    gx = matmul(gy, w.T)
    gw = matmul(x.T, gy)
    return [gx, gw, gb]
  end
end

def linear(x, w, b=nil)
  return Linear.new().call(x, w, b)
end

def linear_simple(x, w, b=nil)
  x = as_variable(x)
  w = as_variable(w)
  t = matmul(x, w)
  if b == nil then
    return t
  end
  y = t + b
  t.data = nil  # Release t.data (ndarray) for memory efficiency
  return y
end


def sigmoid_simple(x)
  np = Numpy
  x = as_variable(x)
  y = as_variable(np.array(1.0)) / (as_variable(np.array(1.0)) + exp(-x))
  return y
end

class Sigmoid < Function
  def forward(x)
    np = Numpy
    # xp = cuda.get_array_module(x)
    # y = 1 / (1 + xp.exp(-x))
    y = np.tanh(x * 0.5) * 0.5 + 0.5  # Better implementation
    return y
  end

  def backward(gy)
    y = self.outputs[0]
    gx = gy * y * (as_variable(1) - y)
    return gx
  end
end

def sigmoid(x)
  return Sigmoid.new().call(x)
end
