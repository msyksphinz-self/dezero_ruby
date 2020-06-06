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
    np = Numpy
    y = np.exp(x)
    return np.array([y])
  end
  def backward(gy)
    np = Numpy
    x = @input.data
    gx = np.exp(x) * gy
    return gx
  end
end

def exp(x)
  return Exp.new().call(x)
end


class Add < Function
  def forward(x0, x1)
    np = Numpy
    @x0_shape = x0.shape()
    @x1_shape = x1.shape()
    y = x0 + x1
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
    y = x0 - x1
    return np.array([y])
  end
  def backward(gy)
    return [gy, gy * -1.0]
  end
end

def sub(x0, x1)
  return Sub.new().call(x0, x1)
end

class Mul < Function
  def forward(x0, x1)
    np = Numpy
    y = x0 * x1
    return np.array([y])
  end
  def backward(gy)
    x0 = @inputs[0]
    x1 = @inputs[1]
    return [gy * x1, gy * x0]
  end
end

def mul(x0, x1)
  x1 = as_array(x1)
  return Mul.new().call(x0, x1)
end


class Div < Function
  def forward(x0, x1)
    np = Numpy
    y = np.array([x0 / x1])
    return y
  end
  def backward(gy)
    x0 = @inputs[0]
    x1 = @inputs[1]
    tmp = [gy / x1, gy * (-x0 / x1 ** 2.0)]
    return tmp
  end
end

def div(x0, x1)
  return Div.new().call(x0, x1)
end


class Neg < Function
  def forward(x)
    np = Numpy
    tmp = -x
    return np.array([tmp])
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
    return np.array([y])
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
    @x_shape = x.shape
    y = x.sum(axis:@axis, keepdims:@keepdims)
    return y
  end

  def backward(gy)
    gy = reshape_sum_backward(gy, @x_shape, @axis, @keepdims)
    gx = broadcast_to(gy, @x_shape)
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
    np = Numpy
    @x_shape = x.shape
    y = util_sum_to(x, @shape)
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
  return SumTo.new(shape).call(x)
end


class MatMul < Function
  def forward(x, w)
    y = x.dot(w)
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
