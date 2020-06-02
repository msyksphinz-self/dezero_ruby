class Square < Function
  def forward(x)
    y = x ** 2
    return y
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
    return y
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
    y = x0 + x1
    return y
  end
  def backward(gy)
    return [gy, gy]
  end
end

def add(x0, x1)
  x1 = as_array(x1)
  return Add.new().call(x0, x1)
end


class Sub < Function
  def forward(x0, x1)
    y = x0 - x1
    return y
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
    y = x0 * x1
    return y
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
    y = x0 / x1
    return y
  end
  def backward(gy)
    x0 = @inputs[0]
    y0 = @inputs[1]
    return [gy / x1, gy * (-x0 / x1 ** 2.0)]
  end
end

def div(x0, x1)
  return Div.new().call(x0, x1)
end


class Neg < Function
  def forward(x)
    tmp = -x
    return tmp
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
    return y
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
    return y
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
