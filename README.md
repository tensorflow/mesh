# Mesh TensorFlow - Model Parallelism Made Easier

[![PyPI
version](https://badge.fury.io/py/mesh-tensorflow.svg)](https://badge.fury.io/py/mesh-tensorflow)
[![GitHub
Issues](https://img.shields.io/github/issues/tensorflow/mesh.svg)](https://github.com/tensorflow/mesh/issues)
[![Contributions
welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/tensorflow/mesh/workflows/build/badge.svg)](https://github.com/tensorflow/mesh/actions?query=workflow%3Abuild)


# Introduction

Mesh TensorFlow (`mtf`) is a language for distributed deep learning, capable of
specifying a broad class of distributed tensor computations.  The purpose of
Mesh TensorFlow is to formalize and implement distribution strategies for your
computation graph over your hardware/processors. For example: "Split the batch
over rows of processors and split the units in the hidden layer across columns
of processors." Mesh TensorFlow is implemented as a layer over TensorFlow.

Watch our [YouTube video](https://www.youtube.com/watch?v=HgGyWS40g-g).


## Do I need Mesh TensorFlow?

If you just want data-parallel training (batch-splitting), then you do not need
Mesh TensorFlow, though Mesh TensorFlow can do this.  The most common reasons
for more sophisticated parallel computation are:

* The parameters of the model do not fit on one device - e.g. a
5-billion-parameter language model.

* An example is so large that the activations do not fit on one device. - e.g.
large 3D image model([`experimental/unet.py`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/unet.py)).

* Lower-latency parallel inference (at batch size 1).

## The Mesh TensorFlow Approach to Distributed Computation

* A "Mesh" is an n-dimensional array of processors, connected by a network.

* Each tensor is distributed (split and/or replicated) across all processors
  in a mesh.

* Tensor dimensions and mesh dimensions are named.  The layouts of all tensors
  follow from a set of user-defined layout rules which specify which
  tensor-dimensions are split across which mesh-dimensions.  This ensures that
  the corresponding dimensions in different tensors are split in the same
  manner.

* Layouts do not affect results - only performance.

* The implementation of an operation involves parallel computation on all
  processors in the mesh, and sometimes also collective communication.  A
  processor usually just manipulates the slices of the input tensors already
  resident on that processor, and produces the slice of the output that goes on
  that processor.

## Getting Started

### Installation

To install the latest stable version, run

```sh
pip install mesh-tensorflow
```

To install the latest development version, run

```sh
pip install -e "git+https://github.com/tensorflow/mesh.git#egg=mesh-tensorflow"
```

Installing `mesh-tensorflow` does not automatically install or update
TensorFlow. We recommend installing it via `pip install tensorflow` or `pip
install tensorflow-gpu`. See TensorFlow’s
[installation instructions for details](https://www.tensorflow.org/install/).
If you're using a development version of Mesh TensorFlow, you may need to
use TensorFlow's nightly package (`tf-nightly`).

### Example Network (MNIST)

To illustrate, let us consider a simple model for the MNIST image-classification
task.  Our network has one hidden layer with 1024 units, and an output layer
with 10 units (corresponding to the 10 digit classes).

The code consists of two parts, the first describing the mathematical
operations, and the second describing the devices and tensor/computation layout.
For the full example, see [`examples/mnist.py`](
https://github.com/tensorflow/mesh/blob/master/examples/mnist.py).
TODO(noam): verify that this code works.

```Python
# tf_images is a tf.Tensor with shape [100, 28, 28] and dtype tf.float32
# tf_labels is a tf.Tensor with shape [100] and dtype tf.int32
graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")
batch_dim = mtf.Dimension("batch", 100)
rows_dim = mtf.Dimension("rows", 28)
cols_dim = mtf.Dimension("cols", 28)
hidden_dim = mtf.Dimension("hidden", 1024)
classes_dim = mtf.Dimension("classes", 10)
images = mtf.import_tf_tensor(
    mesh, tf_images, shape=[batch_dim, rows_dim, cols_dim])
labels = mtf.import_tf_tensor(mesh, tf_labels, [batch_dim])
w1 = mtf.get_variable(mesh, "w1", [rows_dim, cols_dim, hidden_dim])
w2 = mtf.get_variable(mesh, "w2", [hidden_dim, classes_dim])
# einsum is a generalization of matrix multiplication (see numpy.einsum)
hidden = mtf.relu(mtf.einsum(images, w1, output_shape=[batch_dim, hidden_dim]))
logits = mtf.einsum(hidden, w2, output_shape=[batch_dim, classes_dim])
loss = mtf.reduce_mean(mtf.layers.softmax_cross_entropy_with_logits(
    logits, mtf.one_hot(labels, classes_dim), classes_dim))
w1_grad, w2_grad = mtf.gradients([loss], [w1, w2])
update_w1_op = mtf.assign(w1, w1 - w1_grad * 0.001)
update_w2_op = mtf.assign(w2, w2 - w2_grad * 0.001)
```

In the code above, we have built a Mesh TensorFlow graph, which is simply
a Python structure.  We have completely defined the mathematical operations.
In the code below, we specify the mesh of processors and the layout of the
computation.

```Python
devices = ["gpu:0", "gpu:1", "gpu:2", "gpu:3"]
mesh_shape = [("all_processors", 4)]
layout_rules = [("batch", "all_processors")]
mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
    mesh_shape, layout_rules, devices)
lowering = mtf.Lowering(graph, {mesh:mesh_impl})
tf_update_ops = [lowering.lowered_operation(update_w1_op),
                 lowering.lowered_operation(update_w2_op)]
```

The particular layout above implements data-parallelism, splitting the batch of
examples evenly across all four processors.  Any Tensor with a "batch" dimension
(e.g. `images`, `h`, `logits`, and their gradients) is split in that dimension
across all processors, while any tensor without a "batch" dimension (e.g. the
model parameters) is replicated identically on every processor.

Alternatively, for model-parallelism, we can set
`layout_rules=[("hidden", "all_processors")]`.  In this case,
any tensor with a "hidden" dimension (e.g. `hidden`, `w1`, `w2`)  is split,
while any other tensor (e.g. `image`, `logits`) is fully replicated.

We can even combine data-parallelism and model-parallelism on a 2-dimensional
mesh of processors.  We split the batch along one dimension of the mesh, and the
units in the hidden layer along the other dimension of the mesh, as below.  In
this case, the hidden layer is actually tiled between the four processors, being
split in both the "batch" and "hidden_units" dimensions.

```Python
mesh_shape = [("processor_rows", 2), ("processor_cols", 2)]
layout_rules = [("batch", "processor_rows"), ("hidden", "processor_cols")]
```

## Where does the network communication happen?

Some Mesh TensorFlow operations cause network communication.  For example, an
einsum (generalized matrix multiplication) is computed as follows:

* On each processor, compute the einsum of the slices of the two operands that
  are local to that processor.
* If no reduced-out dimensions are split, then we are done.
* If reduced-out dimensions are split, then perform an "allreduce" operation 
  on the resulting slices - summing across any mesh dimensions over which the
  reduced-out dimensions are split.

Where the allreduces happen depends will depend on the computation layout.
For example, in a data-parallel layout where the "batch" dimension is split,
allreduces will happen when computing the parameter gradients, since this
involves matrix multiplications which reduce out the "batch" dimension.

## How do I pick a layout?

While results do not depend on layout (except in the realm of roundoff errors
and random seeds), performance and memory consumption depend heavily on layout.
Fortunately, the auto_mtf subpackage provides a method for automatically
choosing a layout.  For more information about what auto_mtf is doing to choose
a layout, see its [README](mesh_tensorflow/auto_mtf/README.md) file.

```Python
import mesh_tensorflow.auto_mtf

graph = mtf.Graph()
mesh = mtf.Mesh(graph, "my_mesh")
# Insert model code here.
outputs = [logits, loss]  # iterable of mtf.Tensor, the outputs you're computing
mesh_shape = [("processor_rows", 2), ("processor_cols", 2)]
layout_rules = mtf.auto_mtf.layout(graph, mesh_shape, outputs)
```

It is possible for advanced users to eke out additional performance by tuning
the layout (and model) further.  Mesh TensorFlow helps by accumulating and
printing counters of computation/communication.  To start, here are some
tricks/guidelines.

* It is illegal for two dimensions of the same tensor to be split across the
  same mesh dimension.
* For any compute-intense operation (e.g. einsum), make sure that all
  mesh-dimensions are used to split dimensions of the inputs or outputs.
  Otherwise, computation is duplicated.
* To keep the ratio of compute/communication high (i.e. not be bandwidth-bound),
  split dimensions into large chunks.  This should be familiar in the
  data-parallelism case, where we want a large batch size per processor to avoid
  spending most of our time communicating.

# The Mesh TensorFlow Language

Mesh TensorFlow (v0.0) is implemented as a Python library which can generate
part of a TensorFlow graph.  The user first builds a `mtf.Graph` (the analog of
a TensorFlow graph) made up of `mtf.Tensor`s and `mtf.Operation`s.  As in
TensorFlow, this graph consists of simple Python objects.  The user then creates
a `mtf.Lowering` object, which lowers the `mtf.Graph` into TensorFlow, adding to
the default TensorFlow graph.

The Mesh TensorFlow language is nearly identical to TensorFlow, with the
familiar notion of a Graph, Tensors, Operations, and automatic gradient
computation.  The principal differences are as follows:

## Meshes replace devices

A `Mesh` is a n-dimensional array of processors with named dimensions.  Each
`Tensor` is assigned to a `Mesh`, instead of a device.

## Tensor dimensions are named

Each `Tensor` has a static `Shape`, which is a tuple of different "Dimensions".
A `Dimension` is a `(name, size)` pair. For example, the shape of a `Tensor`
representing a batch of images might be:

`[("batch", 100), ("rows", 28"), ("cols", 28), ("channels", 3)]`.

## Layouts

A `Tensor` is laid out on its mesh with one slice on each processor.  A `Tensor`
"layout", is an injective partial map specifying which dimensions of the tensor
are (evenly) split across which dimensions of the mesh.  No dimension of a
tensor may be split across two dimensions of its mesh and no two dimensions of a
tensor may be split across the same dimension of its mesh.  The user defines a
global set of layout rules in the form of (tensor-dimension-name,
mesh-dimension-name) pairs.  A dimension of a tensor is split across a dimension
of its mesh if there is a matching rule.

### Example Layouts

Take our example `Tensor` `image_batch` with shape: 
`[("batch", 100), ("rows", 28"), ("cols", 28), ("channels", 3)]`

Assume that this `Tensor` is assigned to a mesh of 8 processors with shape:
`[("processor_rows", 2), ("processor_cols", 4)]`

* If we use an empty set of layout rules `[]`, we get no splitting.  Each
  processor contains the whole `Tensor`.

* If we use the layout rules `"batch:processor_cols"`, then the `"batch"`
  dimension of the `Tensor` is split across the `"processor_cols"` dimension of
  the batch.  This means that each processor contains a Tensor slice with shape
  `[25, 28, 28, 3]`.  For example, processors (0, 3) and (1, 3) contain
  identical slices - `image_batch[75:100, :, :, :]`.

* If we use the layout rules `"rows:processor_rows;cols:processor_cols"`, 
  then the image is split in two dimensions, with each processor containing one
  spatial tile with shape `[100, 14, 7, 3]`.   For example, processor (0, 1)
  contains the slice `image_batch[:, 0:14, 7:14, :]`.

Some layout rules would lead to illegal layouts:

* `"batch:processor_rows;rows:processor_rows"` is illegal because two tensor
  dimensions could not be split across the same mesh dimension.

* `"channels:processor_rows"` is illegal because the size of the tensor
  dimension is not evenly divisible by the size of the mesh dimension.

## Einsum

Mesh TensorFlow uses Einstein-summation notation, `mtf.einsum(inputs,
output_shape)`, using the (named) `Dimensions` as the symbols.  Matrix
multiplication, broadcast, sum-reduction, and transposition can all be expressed
as special cases of `mtf.einsum`, though the familiar interfaces are also
supported.  The operation is lowered to slice-wise `tf.einsum`s, followed by
allreduce across any mesh-dimensions corresponding to the summed-out Tensor
dimensions.

## Reshape can be expensive

`mtf.reshape(x, new_shape)` is used to change a `Tensor`'s shape, potentially
leading to a new tensor layout and hence network communication.

# CPU/GPU/TPU implementations

Mesh TensorFlow works on CPU, GPU and TPU.  The TPU implementation is very
different from the CPU/GPU implementation.

Multi-CPU/GPU meshes are implemented with `PlacementMeshImpl`.  In this case
Mesh TensorFlow emits separate TensorFlow operations placed on the different
devices, all in one big TensorFlow graph.

TPU meshes are implemented in with `SimdMeshImpl`.  In this case,
Mesh TensorFlow emits TensorFlow operations (and communication collectives) from
the perspective of one core, and this same program runs on every core, relying
on the fact that each core actually performs the same operations.  This
piggy-backs on the TPU data-parallelism infrastructure, which operates the same
way.  This "SIMD" approach keeps the TensorFlow and XLA graphs from growing with
the number of cores.  The differences between cores are as follows:

* different slices of the variables (this works now)
* different positions in the collective communication (this works now)
* different slices of the infed and outfed tensors.  We currently work around
  this by requiring that all imported/exported tensors be fully-replicated.  In
  the future, we should handle this correctly.

# Experimental features

The input pipeline of Mesh Tensorflow models might become a bottleneck, when
training with large input (e.g., high resolution images). We provide new APIs
and a new input pipeline for you to run Mesh Tensorflow models. You can find
them under the [`experimental/`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/)
folder. We suggest that you give them a try when your input is so large that
running Mesh Tensorflow models with the default APIs is almost infeasible.
To be more specific:

* The BROADCAST mode in TPUEstimator does not scale up to large inputs (images
  of tens of millions of pixels). We provide a new input pipeline:
  [`experimental/input_reader.py`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/input_reader.py).
  See [`experimental/model_executor.py`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/model_executor.py)
  on how to use it.
* If your model takes images as input and has convolution layers. You cannot
  directly map image height and width dimensions to mesh dimensions, due to the
  sliding-window nature of convolution. Instead, you should use spatial
  partitioning. We provide examples in
  [`experimental/unet.py`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/unet.py).
* If you want more control on the training and evaluation loop, instead of using
  the default API (TPUEstimator) to run your model, you can use low level APIs
  in [`experimental/model_executor.py`](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/experimental/model_executor.py).

Note that we did not test the experimental code on GPUs. We ran them on TPUs.
We believe that some debugging would be required for it to work on GPUs.

# Instructions for running on cloud-tpu

Note: It requires `tensorflow>=1.11.0`.

## Prerequisite

Please go through the
[Transformer tutorial](https://cloud.google.com/tpu/docs/tutorials/transformer).

## Create VM and TPU instance in Cloud console

TODO(trandustin,ylc): update given mtf pypi package

```sh
ctpu up -name=ylc-mtf-donut -tf-version=nightly -tpu-size=v2-8 -zone=us-central1-b
```

## SSH into VM

```sh
git clone https://github.com/tensorflow/mesh.git
cd mesh/
pip install --user .
```

## Run the Transformer model (no Tensor2Tensor dependencies)

```sh
pip install tensorflow_datasets

cd mesh/
DATA_DIR=gs://noam-mtf/data
MODEL_DIR=gs://noam-mtf/transformer_standalone
TPU=noam-mtf-donut

# MODEL HPARAMS AND DIRECTORY  (uncomment one)
# base model
MODEL=./transformer/gin/model_base.gin
# 5B parameters (too big for this dataset, only trains with model-parallelism)
# MODEL=./transformer/gin/model_5b.gin

# UNCOMMENT ONE OF THESE
# Data-parallelism
LAYOUT=./transformer/gin/layout_data_parallel.gin
# Model-parallelism
# LAYOUT=./transformer/gin/layout_model_parallel.gin
# Data-parallelism and Model-Parallelism
# LAYOUT=./transformer/gin/layout_data_and_model_parallel.gin

# TRAIN
python examples/transformer_standalone.py \
  --tpu=$TPU --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gin_file=$MODEL \
  --gin_file=$LAYOUT --gin_param="run.mode='train'"

# EVAL
python examples/transformer_standalone.py \
  --tpu=$TPU --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gin_file=$MODEL \
  --gin_file=$LAYOUT --gin_param="run.mode='evaluate'"
```

The above code will train on the LM1B language modeling benchmark, as specified
in `examples/transformer_standalone_defaults.gin`. To train a
sequence-to-sequence model on WMT14 en-de, change `utils.run.dataset` to
`wmt_translate_ende/ende_subwords8k_t2t` and set `utils.run.mode` to `True`.
Note that the `wmt_translate_ende/ende_subwords8k_t2t` dataset was removed from
TensorFlow Datasets in
[commit 211cb6f](https://github.com/tensorflow/datasets/commit/211cb6f082c5cc3c482e37d70234142a8fda2db3),
so in order to train a model using this dataset you need to install a version of
TFDS before this commit. Then, you can decode the WMT en-de development set
and evaluate it using [SacreBLEU](https://github.com/mjpost/sacreBLEU) like so:

```
# INFER
pip3 install sacrebleu
mkdir ~/input ~/output
DECODE_INPUT=/home/$USER/input/ende.dev
DECODE_OUTPUT=/home/$USER/output/ende.dev.out
~/.local/bin/sacrebleu -t wmt13 -l en-de --echo src > $DECODE_INPUT
python examples/transformer_standalone.py \
  --tpu=$TPU --data_dir=$DATA_DIR --model_dir=$MODEL_DIR --gin_file=$MODEL \
  --gin_file=$LAYOUT \
  --gin_param="decode_from_file.input_filename='$DECODE_INPUT'" \
  --gin_param="decode_from_file.output_filename='$DECODE_OUTPUT'" \
  --gin_param="run.mode='infer'"

# Compute BLEU score for dev set
cat $DECODE_OUTPUT | ~/.local/bin/sacrebleu -t wmt13 -l en-de -tok intl
```


## Run the Transfomer model with Tensor2Tensor config
```sh
git clone https://github.com/tensorflow/tensor2tensor.git
cd tensor2tensor/
pip install --user  .
```

Before running the model, you need to prepare the training data and bucket for
storing checkpoints. Refer to the
[Transformer tutorial](https://cloud.google.com/tpu/docs/tutorials/transformer)
to learn how to generate the training data and create buckets.

```sh
CONF=mtf_transformer_paper_tr_0_mesh_8
NAME=ende_$CONF\_0828
MODEL=mtf_transformer
PROBLEM=translate_ende_wmt32k_packed

DATA_DIR=gs://xxxx
OUT_DIR=gs://xxxx
TPU_NAME=ylc-mtf-donut

tensor2tensor/bin/t2t-trainer \
  --model=$MODEL \
  --hparams_set=$CONF \
  --problem=$PROBLEM \
  --train_steps=10000 \
  --eval_steps=200 \
  --data_dir=$DATA_DIR \
  --output_dir=$OUT_DIR \
  --use_tpu=True \
  --cloud_tpu_name=$TPU_NAME
```


## Run the toy model without Tensor2Tensor dependencies

  This toy model contains two fully-connected layers which aim to train a
  identity function: f(x) = x. Since there are 8 TPU cores, we can arbitrary
  change the FLAGS.mesh_shape and FLAGS.layout to achieve different
  data-parallelism and model-parallelism strategies.

```sh
MODEL_DIR=gs://xxxx
TPU_NAME=ylc-mtf-donut

# 2 ways data-parallelism and 4 ways model-parallelism.
# In this configuration, we split the batch dimension into 2 cores and the
# hidden dimension into 4 cores.
python examples/toy_model_tpu.py \
  --tpu=$TPU \
  --model_dir=$MODEL_DIR \
  --io_size=8 \
  --hidden_size=8 \
  --mesh_shape='x:2;y:4' \
  --layout='batch:x;hidden:y'

# 8 ways model-parallelism.
# In this configuration, We split the hidden dimension into 8 cores.
python examples/toy_model_tpu.py \
  --tpu=$TPU \
  --model_dir=$MODEL_DIR \
  --io_size=8 \
  --hidden_size=8 \
  --mesh_shape='all:8' \
  --layout='hidden:all'
```

## References

> N. Shazeer, Y. Cheng, N. Parmar, D. Tran, A. Vaswani, P. Koanantakool,
> P. Hawkins, H. Lee, M. Hong, C. Young, R. Sepassi, and B. Hechtman.
> [Mesh-TensorFlow: Deep learning for supercomputers.](https://arxiv.org/abs/1811.02084)
> In _Neural Information Processing Systems_, 2018.

```none
@inproceedings{shazeer2018mesh,
  author = {Noam Shazeer and Youlong Cheng and Niki Parmar and Dustin Tran and Ashish Vaswani and Penporn Koanantakool and Peter Hawkins and HyoukJoong Lee and Mingsheng Hong and Cliff Young and Ryan Sepassi and Blake Hechtman},
  title = {{Mesh-TensorFlow}: Deep Learning for Supercomputers},
  booktitle = {Neural Information Processing Systems},
  year = {2018},
}
```
