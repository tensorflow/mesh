import mesh_tensorflow as mtf
from mesh_tensorflow.signal_ops import fft3d, ifft3d
import tensorflow as tf


class FFTTest(tf.test.TestCase):
  def setUp(self):
    super(FFTTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, "my_mesh")
    volume_size = 32
    batch_dim = mtf.Dimension("batch", 1)
    slices_dim = mtf.Dimension("slices", volume_size)
    rows_dim = mtf.Dimension("rows", volume_size)
    cols_dim = mtf.Dimension("cols", volume_size)
    self.shape = [batch_dim, slices_dim, rows_dim, cols_dim,]
    volume_shape = [d.size for d in self.shape]
    self.volume = tf.complex(
        tf.random.normal(volume_shape),
        tf.random.normal(volume_shape),
    )
    self.volume_mesh = mtf.import_tf_tensor(self.mesh, self.volume, shape=self.shape)


  def testFft3d(self):
    outputs = fft3d(self.volume_mesh, freq_dims=self.shape[1:4])
    assert len(outputs.shape) == 4
    assert outputs.dtype == tf.complex64
    # assert outputs.shape == mtf.Shape(self.shape)
    assert [d.size for d in outputs.shape] == [d.size for d in self.shape]
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    outputs_tf = lowering.export_to_tf_tensor(outputs)
    expected_outputs = tf.signal.fft3d(self.volume)
    expected_outputs = tf.transpose(expected_outputs, perm=[0, 2, 3, 1])
    self.assertAllClose(
      outputs_tf,
      expected_outputs,
      rtol=1e-4,
      atol=1e-4,
    )

  def testIfft3d(self):
    outputs = ifft3d(
        self.volume_mesh,
        # ordering is not the same for ifft3d
        dims=[self.shape[3], self.shape[1], self.shape[2]],
    )
    assert len(outputs.shape) == 4
    assert outputs.dtype == tf.complex64
    # assert outputs.shape == mtf.Shape(self.shape)
    assert [d.size for d in outputs.shape] == [d.size for d in self.shape]
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
      shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    outputs_tf = lowering.export_to_tf_tensor(outputs)
    expected_outputs = tf.signal.ifft3d(self.volume)
    expected_outputs = tf.transpose(expected_outputs, perm=[0, 3, 1, 2])
    self.assertAllClose(
      outputs_tf,
      expected_outputs,
      rtol=1e-4,
      atol=1e-4,
    )
