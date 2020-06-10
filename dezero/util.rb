def _dot_var (v, verbose=false)

  # puts "v.name = " + v.name.to_s

  name = v.name == nil ? '' : v.name
  if verbose and v.data != nil then
    if v.name != nil then
      name = name + ': '
    end
    puts "v.shape = " + v.class.to_s
    name += v.shape.to_s + ' ' + v.dtype.to_s
  end
  return sprintf("%s [label=\"%s\", color=orange, style=filled]\n",
                 v.object_id, name)
end

def _dot_func(f)

  txt = sprintf("%s [label=\"%s\", color=lightblue, style=filled, shape=box]\n",
                f.object_id, f.class.to_s)
  f.inputs.each{|x|
    txt += sprintf("%s -> %s\n", x.object_id, f.object_id)
  }
  f.outputs.each{|y|
    txt += sprintf("%s -> %s\n", f.object_id, y.__getobj__.object_id)
  }
  return txt
end


def get_dot_graph(output, verbose=true)
  funcs = Array.new()
  seen_set = Set.new()
  txt = ""

  def add_func(f, funcs, seen_set)
    if not seen_set.include?(f) then
      funcs.push(f)
      seen_set.add(f)
    end
  end

  add_func(output.creator, funcs, seen_set)
  txt += _dot_var(output, verbose)

  while not funcs.empty? do
    func = funcs.pop
    txt += _dot_func(func)
    func.inputs.each{|x|
      txt += _dot_var(x, verbose)

      if x.creator != nil then
        add_func(x.creator, funcs, seen_set)
      end
    }
  end

  return "digraph g {\n" + txt + "}"
end


def plot_dot_graph(output, verbose=true, to_file='graph.png')
  dot_graph = get_dot_graph(output, verbose)

  f = File.new("tmp_graph.dot", "w")
  f.write(dot_graph)

  cmd = "dot " + "tmp_graph.dot" + " -T png -o " + to_file
  result = %x[#{cmd}]
  puts cmd
  puts result
end


# =============================================================================
# Utility functions for numpy (numpy magic)
# =============================================================================
def util_sum_to(x, shape)
  ndim = PyCall::eval("len(#{shape})")
  lead = x.ndim - ndim
  # puts "lead = " + lead.to_s + ", shape = " + shape.to_s + ", " + ndim.to_s
  lead_axis = PyCall::eval("tuple(range(#{lead}))")

  axis = PyCall::eval("tuple([i + #{lead} for i, sx in enumerate(#{shape}) if sx == 1])")
  y = x.sum(lead_axis + axis, keepdims:true)
  # puts "  Util_sum_to.shape = " + y.shape.to_s
  if lead > 0 then
    np = Numpy
    # puts "  y = " + y.class.to_s + ", " + lead_axis.to_s
    y = y.squeeze(lead_axis)
  end
  # puts "  Util_sum_to.shape = " + y.shape.to_s
  return y
end


def reshape_sum_backward(gy, x_shape, axis, keepdims)
  ndim = PyCall::len(x_shape)
  tupled_axis = axis
  if axis == nil then
    tupled_axis = nil
  elsif not PyCall::eval("hasattr(#{axis}, 'len')") then
    tupled_axis = [axis]
  end

  shape = []
  if not (ndim == 0 or tupled_axis == nil or keepdims) then
    actual_axis = tupled_axis.each{|a|
      a >= 0 ? a : a + ndim
    }
    shape = [gy.shape]
    for a in PyCall::eval("sorted(#{actual_axis})") do
      shape.insert(a, 1)
    end
  else
    shape = gy.shape
  end

  gy = gy.reshape(shape)  # reshape
  return gy
end

def mean_squared_error(x0, x1)
  np = Numpy
  diff = x0 - x1
  return sum(diff ** 2) / as_variable(np.array(diff.size))
end
