# Auto Mesh TensorFlow

Auto Mesh TensorFlow (`auto_mtf`) is a utility to help choose a good Mesh
TensorFlow layout for your computation.

## Auto MTF Usage Example

```Python
import mesh_tensorflow.auto_mtf

graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")
# Insert model code here.
outputs = [logits, loss]  # iterable of mtf.Tensor, the outputs you're computing
mesh_shape = [("processor_rows", 2), ("processor_cols", 2)]
layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, outputs)
```

## How does Auto MTF work?

Auto MTF currently tries to choose the layout which minimizes the peak memory
usage of the computation.  In the future, we hope to also support layouts which
result in efficient computations while keeping peak memory under a threshold.
Currently, Auto MTF works by writing this search for the memory-minimizing legal
layout as a
[Mixed-Integer Program](https://en.wikipedia.org/wiki/Integer_programming).
There are several ingredients that go into the construction of this
mixed-integer program:

* A schedule to compute the operations in (affects the peak memory usage).
* An estimate of the size of each tensor.
* Rules to determine which layouts are legal.

Currently, the schedule is based on a greedy list scheduler heuristic and the
size estimates are based on the Mesh TensorFlow tensors.  It is possible to
improve the quality of the produced layouts by improving the quality of these
inputs to the mixed-integer program.
