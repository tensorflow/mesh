# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of beam search with penalties."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from mesh_tensorflow import ops_with_redefined_builtins as mtf
import tensorflow.compat.v1 as tf

# Assuming EOS_ID is 1
EOS_ID = 1
# Default value for INF
INF = 1. * 1e7


def compute_topk_scores_and_seq(sequences, scores, scores_to_gather, flags,
                                beam_dim, prefix="default"):
  """Given sequences and scores, will gather the top k=beam size sequences.

  This function is used to grow alive, and finished. It takes sequences,
  scores, and flags, and returns the top k from sequences, scores_to_gather,
  and flags based on the values in scores.

  This method permits easy introspection using tfdbg.  It adds two named ops
  that are prefixed by `prefix`:
    - _topk_seq: the tensor for topk_seq returned by this method.
    - _topk_flags: the tensor for topk_finished_flags returned by this method.

  Args:
    sequences: Tensor of sequences that we need to gather from.
      [batch_size, beam_size, seq_length]
    scores: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will use these to compute the topk.
    scores_to_gather: Tensor of scores for each sequence in sequences.
      [batch_size, beam_size]. We will return the gathered scores from here.
      Scores to gather is different from scores because for grow_alive, we will
      need to return log_probs, while for grow_finished, we will need to return
      the length penalized scores.
    flags: Tensor of bools for sequences that say whether a sequence has reached
      EOS or not
    beam_dim: mtf.Dimension
    prefix: an optional string
  Returns:
    Tuple of
    (topk_seq [batch_size, beam_size, decode_length],
     topk_gathered_scores [batch_size, beam_size],
     topk_finished_flags[batch_size, beam_size],
     selector)
  """
  unused_batch_dim, old_beam_dim, unused_length_dim = sequences.shape.dims
  _, topk_indices = mtf.top_k(scores, old_beam_dim, k_dim=beam_dim)

  selector = mtf.one_hot(topk_indices, old_beam_dim, dtype=tf.float32)

  # Gather up the highest scoring sequences.
  # For each operation added, give it
  # a concrete name to simplify observing these operations with tfdbg.
  # Clients can capture these tensors by watching these node names.
  def gather(tensor, name):
    with tf.name_scope(prefix + name):
      output_shape = mtf.Shape(
          [beam_dim if d == old_beam_dim else d for d in tensor.shape.dims])
      return mtf.gather(
          tensor, topk_indices, old_beam_dim, output_shape=output_shape)
  topk_seq = gather(sequences, "_seq")
  topk_flags = gather(flags, "_flags")
  topk_gathered_scores = gather(scores_to_gather, "_scores")
  return topk_seq, topk_gathered_scores, topk_flags, selector


@gin.configurable
def beam_search(logits_fn,
                initial_ids,
                alpha,
                states=None,
                eos_id=EOS_ID,
                stop_early=True,
                decode_length=None,
                use_tpu=True,
                dtype=tf.float32,
                layout=None,
                mesh_shape=None,
                num_prefilter=2):
  """Beam search with length penalties.

  Requires a function that can take the currently decoded symbols and return
  the logits for the next symbol. The implementation is inspired by
  https://arxiv.org/abs/1609.08144.

  When running, the beam search steps can be visualized by using tfdbg to watch
  the operations generating the output ids for each beam step.  These operations
  have the pattern:
    (alive|finished)_topk_(seq,scores)

  Operations marked `alive` represent the new beam sequences that will be
  processed in the next step.  Operations marked `finished` represent the
  completed beam sequences, which may be padded with 0s if no beams finished.

  Operations marked `seq` store the full beam sequence for the time step.
  Operations marked `scores` store the sequence's final log scores.

  The beam search steps will be processed sequentially in order, so when
  capturing observed from these operations, tensors, clients can make
  assumptions about which step is being recorded.

  num_prefilter is a theoretically lossy shortcut around slow performance of
  top_k on TPU on large Tensors and large k.  This option should be removed once
  better top_k implementations on TPU are avialable.  If num_prefilter is set to
  a nonzero value, then at each step we first compute the top num_prefilter
  sequences per beam and then compute the top k sequences overall from among
  those.  Empirically, there seems to be no quality difference in setting
  num_prefilter to 2.

  Args:
    logits_fn: Interface to the model, to provide logits.
        Should take:
          step_num - mtf Scalar
          ids - mtf Tensor with shape [batch, beam, length]
        Should return:
          logits - [batch, beam, vocab_size], dtype=dtype
    initial_ids: a mtf.Tensor with shape [batch_dim, beam_dim, length_dim])
    alpha: alpha for length penalty.
    states: list of mtf.Tensor
    eos_id: ID for end of sentence.
    stop_early: a boolean - stop once best sequence is provably determined.
    decode_length: a mtf Scalar of dtype tf.int32 - maximum length of decodes
    use_tpu: a boolean
    dtype: a tf.dtype
    layout: an optional string
    mesh_shape: an optional string
    num_prefilter: an optional integer
  Returns:
    Tuple of
    (decoded beams [batch, beam, length]
     decoding probabilities [batch, beam_size])
  """
  batch_dim, beam_dim, length_dim = initial_ids.shape.dims
  batch_and_beam_dim = mtf.Dimension(
      batch_dim.name, batch_dim.size * beam_dim.size)
  mesh = initial_ids.mesh

  batch_by_beam = mtf.Shape([batch_dim, beam_dim])
  initial_log_probs = mtf.broadcast(
      mtf.one_hot(
          mtf.constant(mesh, 0, dtype=tf.int32),
          beam_dim,
          on_value=0.0,
          off_value=-INF,
          dtype=dtype),
      batch_by_beam)

  length_scalar = mtf.constant(mesh, length_dim.size, dtype=tf.int32)
  if decode_length is None:
    decode_length = length_scalar
  else:
    decode_length = mtf.minimum(decode_length, length_scalar)

  alive_log_probs = initial_log_probs
  alive_seq = initial_ids

  # Finished will keep track of all the sequences that have finished so far
  # Finished log probs will be negative infinity in the beginning
  # finished_flags will keep track of booleans
  finished_seq = initial_ids
  finished_scores = mtf.constant(mesh, -INF, batch_by_beam, dtype=dtype)

  # Setting the scores of the initial to negative infinity.
  finished_flags = mtf.constant(mesh, False, batch_by_beam, tf.bool)

  def grow_finished(finished_seq, finished_scores, finished_flags, curr_seq,
                    curr_scores, curr_finished):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      finished_seq: Current finished sequences.
        [batch, beam, length]
      finished_scores: scores for each of these sequences.
        [batch, beam]
      finished_flags: finished bools for each of these sequences.
        [batch, beam]
      curr_seq: current topk sequence that has been grown by one position.
        [batch, beam, length]
      curr_scores: scores for each of these sequences. [batch, beam]
      curr_finished: Finished flags for each of these sequences.
        [batch, beam]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences,
         None (no states))
    """

    # Set the scores of the unfinished seq in curr_seq to large negative
    # values
    curr_scores += (1. - mtf.cast(curr_finished, curr_scores.dtype)) * -INF
    unused_batch_dim, beam_dim, unused_length_dim = finished_seq.shape.dims
    # concatenating the sequences and scores along beam axis
    def _my_concat(a, b):
      a = mtf.rename_dimension(a, "beam", "triple_beam")
      b = mtf.rename_dimension(b, "double_beam", "triple_beam")
      return mtf.concat([a, b], "triple_beam")

    curr_finished_seq = _my_concat(finished_seq, curr_seq)
    curr_finished_scores = _my_concat(finished_scores, curr_scores)
    curr_finished_flags = _my_concat(finished_flags, curr_finished)
    return compute_topk_scores_and_seq(
        curr_finished_seq, curr_finished_scores, curr_finished_scores,
        curr_finished_flags, beam_dim, "grow_finished")

  def grow_alive(curr_seq, curr_scores, curr_log_probs, curr_finished):
    """Given sequences and scores, will gather the top k=beam size sequences.

    Args:
      curr_seq: current topk sequence that has been grown by one position.
        [batch, beam, length]
      curr_scores: scores for each of these sequences. [batch_size, beam_size]
      curr_log_probs: log probs for each of these sequences.
        [batch, beam]
      curr_finished: Finished flags for each of these sequences.
        [batch, beam]
    Returns:
      Tuple of
        (Topk sequences based on scores,
         log probs of these sequences,
         Finished flags of these sequences)
    """
    # Set the scores of the finished seq in curr_seq to large negative
    # values
    curr_scores += mtf.cast(curr_finished, curr_scores.dtype) * -INF
    return compute_topk_scores_and_seq(curr_seq, curr_scores, curr_log_probs,
                                       curr_finished, beam_dim,
                                       "grow_alive")

  def grow_topk(i, alive_seq, alive_log_probs, states=None):
    r"""Inner beam search loop.

    This function takes the current alive sequences, and grows them to topk
    sequences where k = 2*beam. We use 2*beam because, we could have beam_size
    number of sequences that might hit <EOS> and there will be no alive
    sequences to continue. With 2*beam_size, this will not happen. This relies
    on the assumption the vocab size is > beam size. If this is true, we'll
    have at least beam_size non <EOS> extensions if we extract the next top
    2*beam words.
    Length penalty is given by = (5+len(decode)/6) ^ -\alpha. Pls refer to
    https://arxiv.org/abs/1609.08144.

    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch, beam, length]
      alive_log_probs: probabilities of these sequences. [batch, beam]
      states: optional list of mtf.Tensor
    Returns:
      Tuple of
        (Topk sequences extended by the next word,
         The log probs of these sequences,
         The scores with length penalty of these sequences,
         Flags indicating which of these sequences have finished decoding,
         list of transformed decoding states)
    """
    logits, new_states = logits_fn(i, alive_seq, states)
    batch_dim, beam_dim, vocab_dim = logits.shape.dims

    # Convert logits to normalized log probs
    candidate_log_probs = mtf.log_softmax(logits, vocab_dim)

    # Multiply the probabilities by the current probabilities of the beam.
    # (batch_size, beam_size, vocab_size) + (batch_size, beam_size, 1)
    log_probs = candidate_log_probs + alive_log_probs

    length_penalty = mtf.pow(((5. + mtf.cast(i + 1, logits.dtype)) / 6.), alpha)

    # scores have shape [batch, beam, vocab]
    curr_scores = log_probs / length_penalty

    # We find the top 2k sequences to make sure we get k alive sequences.
    #
    # TODO(noam): This is inefficient.  We should separately compute the k
    # finished sequences (previously alive sequences + EOS), and the top k new
    # alive sequences.
    double_beam = mtf.Dimension("double_beam", beam_dim.size * 2)

    if use_tpu and layout is not None and mesh_shape is not None:
      # Do some partial top-k-ing first locally to avoid communication.
      # We reshape the logits from:
      #   [batch, beam, vocab] to
      #   [batch, beam, major_vocab, minor_vocab]
      # We first reduce (locally) across the minor_vocab dimension.  This makes
      # the thing we need to broadcast smaller.
      # This also enables our shortcut of only picking the top num_prefilter
      #   sequences per beam per major_vocab in the first pass.
      major_vocab_size = mtf.tensor_dim_to_mesh_dim_size(
          layout, mesh_shape, vocab_dim)
      major_vocab = mtf.Dimension(vocab_dim.name, major_vocab_size)
      minor_vocab = mtf.Dimension(
          "minor_vocab", vocab_dim.size // major_vocab_size)
      curr_scores = mtf.reshape(
          curr_scores, [batch_dim, beam_dim, major_vocab, minor_vocab])
      prefilter = mtf.Dimension("prefilter", num_prefilter or double_beam.size)
      # shape = [batch_dim, beam_dim, major_vocab, prefilter]
      top_scores, top_minor_vocab_ids = mtf.top_k(
          curr_scores, reduced_dim=minor_vocab, k_dim=prefilter)
      combined = mtf.Dimension(
          "combined", beam_dim.size * major_vocab.size * prefilter.size)
      top_scores = mtf.reshape(top_scores, [batch_dim, combined])
      top_minor_vocab_ids = mtf.reshape(
          top_minor_vocab_ids, [batch_dim, combined])
      # shpae = [batch_dim, double_beam]
      # ids are indices representing (beam, major_vocab, prefilter)
      top_scores, top_combined_ids = mtf.top_k(
          top_scores, reduced_dim=combined, k_dim=double_beam)
      top_minor_vocab_ids = mtf.gather(
          top_minor_vocab_ids, top_combined_ids, combined,
          output_shape=[batch_dim, double_beam])
      top_beam_index = top_combined_ids // (major_vocab.size * prefilter.size)
      top_combined_ids -= top_beam_index * (major_vocab.size * prefilter.size)
      top_major_vocab_ids = top_combined_ids // prefilter.size
      top_combined_ids -= top_major_vocab_ids * prefilter.size
      top_ids = top_major_vocab_ids * minor_vocab.size + top_minor_vocab_ids
    else:
      beam_and_vocab_dim = mtf.Dimension(
          "beam_and_vocab", beam_dim.size * vocab_dim.size)
      flat_shape = mtf.Shape([batch_dim, beam_and_vocab_dim])
      # Flatten out (beam_size, vocab_size) probs into a list of possibilities
      flat_curr_scores = mtf.reshape(
          curr_scores, flat_shape, name="flatten_scores")
      top_scores, top_ids = mtf.top_k(
          flat_curr_scores, reduced_dim=beam_and_vocab_dim, k_dim=double_beam)
      # Work out what beam the top probs are in.
      top_beam_index = top_ids // vocab_dim.size
      top_ids %= vocab_dim.size  # Unflatten the ids

    # Recovering the log probs because we will need to send them back
    top_log_probs = top_scores * length_penalty

    selector = mtf.one_hot(top_beam_index, beam_dim, dtype=tf.float32)

    def my_gather(tensor):
      return mtf.gather(
          tensor, top_beam_index, beam_dim,
          output_shape=mtf.Shape(
              [double_beam if d == beam_dim else d for d in tensor.shape.dims]))

    # Gather up the most probable 2*beams both for the ids and finished_in_alive
    # bools
    top_seq = my_gather(alive_seq)

    # Append the most probable alive
    top_seq += top_ids * mtf.one_hot(i, length_dim, dtype=tf.int32)
    top_finished = mtf.equal(top_ids, eos_id)

    return (
        top_seq, top_log_probs, top_scores, top_finished, new_states, selector)

  def inner_loop(i, alive_seq, alive_log_probs, finished_seq, finished_scores,
                 finished_flags, *states):
    """Inner beam search loop.

    There are three groups of tensors, alive, finished, and topk.
    The alive group contains information about the current alive sequences
    The topk group contains information about alive + topk current decoded words
    the finished group contains information about finished sentences, that is,
    the ones that have decoded to <EOS>. These are what we return.
    The general beam search algorithm is as follows:
    While we haven't terminated (pls look at termination condition)
      1. Grow the current alive to get beam*2 topk sequences
      2. Among the topk, keep the top beam_size ones that haven't reached EOS
      into alive
      3. Among the topk, keep the top beam_size ones have reached EOS into
      finished
    Repeat
    To make things simple with using fixed size tensors, we will end
    up inserting unfinished sequences into finished in the beginning. To stop
    that we add -ve INF to the score of the unfinished sequence so that when a
    true finished sequence does appear, it will have a higher score than all the
    unfinished ones.

    Args:
      i: loop index
      alive_seq: Topk sequences decoded so far [batch_size, beam_size, i+1]
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_seq: Current finished sequences.
        [batch_size, beam_size, i+1]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_flags: finished bools for each of these sequences.
        [batch_size, beam_size]
      *states: mtf Tensors

    Returns:
      Tuple of
        (Incremented loop index
         New alive sequences,
         Log probs of the alive sequences,
         New finished sequences,
         Scores of the new finished sequences,
         Flags indicating which sequence in finished as reached EOS,
         dict of final decoding states)
    """
    states = [mtf.replace_dimensions(
        state, batch_and_beam_dim, [batch_dim, beam_dim]) for state in states]
    # Each inner loop, we carry out three steps:
    # 1. Get the current topk items.
    # 2. Extract the ones that have finished and haven't finished
    # 3. Recompute the contents of finished based on scores.
    (top2k_seq, top2k_log_probs, top2k_scores, top2k_finished,
     new_states, first_selector) = grow_topk(
         i, alive_seq, alive_log_probs, states)
    with tf.variable_scope("grow_alive"):
      alive_seq, alive_log_probs, _, second_selector = grow_alive(
          top2k_seq, top2k_scores, top2k_log_probs, top2k_finished)
    with tf.variable_scope("grow_finished"):
      finished_seq, finished_scores, finished_flags, _ = grow_finished(
          finished_seq, finished_scores, finished_flags, top2k_seq,
          top2k_scores, top2k_finished)
    old_beam_dim = mtf.Dimension("old_beam", beam_dim.size)
    selector = mtf.einsum(
        [mtf.rename_dimension(first_selector, beam_dim.name, old_beam_dim.name),
         second_selector],
        output_shape=[batch_dim, old_beam_dim, beam_dim])
    gathered_states = []
    if use_tpu and layout is not None and mesh_shape is not None:
      # This hack combines the beam dimension with some of the batch dimension.
      # It makes gathering faster on TPU.
      #
      # Instead of multiplying by a [beam, beam] selector matrix, we instead
      # multiply by a [minor_batch*beam, minor_batch*beam] selector matrix.
      # This is theoretically more FLOPs, but it brings the matrix size closer
      # to the magic optimal value of 128.
      #
      # TODO(noam): file a bug with the XLA team to do this automatically
      major_batch_size = mtf.tensor_dim_to_mesh_dim_size(
          layout, mesh_shape, batch_dim)
      major_batch = mtf.Dimension(batch_dim.name, major_batch_size)
      minor_batch = mtf.Dimension(
          "minor_batch", batch_dim.size // major_batch.size)
      old_minor_batch = mtf.Dimension("old_minor_batch", minor_batch.size)
      old_combined = mtf.Dimension(
          "old_combined", minor_batch.size * beam_dim.size)
      combined = mtf.Dimension(
          "new_combined", old_combined.size)
      same_minor_batch = mtf.to_float(
          mtf.equal(mtf.range(mesh, old_minor_batch, tf.float32),
                    mtf.range(mesh, minor_batch, tf.float32)))
      selector = mtf.reshape(
          selector, [major_batch, minor_batch, old_beam_dim, beam_dim])
      selector = mtf.einsum(
          [selector, same_minor_batch],
          output_shape=[major_batch,
                        old_minor_batch, old_beam_dim,
                        minor_batch, beam_dim],
          reduced_dims=[])
      selector = mtf.reshape(selector, [major_batch, old_combined, combined])
      for state in new_states:
        s = mtf.replace_dimensions(
            state, [batch_dim, beam_dim], [major_batch, old_combined])
        s = mtf.einsum(
            [s, mtf.cast(selector, state.dtype)],
            reduced_dims=[old_combined],
            output_shape=mtf.replace_dimensions(
                state.shape, [batch_dim, beam_dim],
                [major_batch, combined]))
        gathered_states.append(mtf.replace_dimensions(
            s, [major_batch, combined], batch_and_beam_dim))
    else:
      for state in new_states:
        state = mtf.einsum(
            [mtf.rename_dimension(state, beam_dim.name, old_beam_dim.name),
             mtf.cast(selector, state.dtype)],
            reduced_dims=[old_beam_dim], output_shape=state.shape)
        state = mtf.replace_dimensions(
            state, [batch_dim, beam_dim], batch_and_beam_dim)
        gathered_states.append(state)

    return (i + 1, alive_seq, alive_log_probs, finished_seq, finished_scores,
            finished_flags) + tuple(gathered_states)

  def _is_finished(i, unused_alive_seq, alive_log_probs, unused_finished_seq,
                   finished_scores, finished_in_finished, *unused_states):
    """Checking termination condition.

    We terminate when we decoded up to decode_length or the lowest scoring item
    in finished has a greater score that the highest prob item in alive divided
    by the max length penalty

    Args:
      i: loop index
      alive_log_probs: probabilities of the beams. [batch_size, beam_size]
      finished_scores: scores for each of these sequences.
        [batch_size, beam_size]
      finished_in_finished: finished bools for each of these sequences.
        [batch_size, beam_size]

    Returns:
      Bool.
    """
    # TODO(noam): support a different decode length...
    # decode_length = mtf.constant(mesh, length_dim.size, dtype=tf.int32)

    # del alive_log_probs, finished_scores, finished_in_finished
    # return mtf.less(i, length_dim.size)
    if not stop_early:
      return mtf.less(i, decode_length)
    max_length_penalty = mtf.pow(
        ((5. + mtf.cast(decode_length, finished_scores.dtype)) / 6.), alpha)
    # The best possible score of the most likely alive sequence.
    lower_bound_alive_scores = mtf.gather(
        alive_log_probs, mtf.constant(mesh, 0, dtype=tf.int32),
        beam_dim) / max_length_penalty

    # Now to compute the lowest score of a finished sequence in finished
    # If the sequence isn't finished, we multiply it's score by 0. since
    # scores are all -ve, taking the min will give us the score of the lowest
    # finished item.
    lowest_score_of_finished_in_finished = mtf.reduce_min(
        finished_scores * mtf.cast(finished_in_finished, finished_scores.dtype),
        reduced_dim=beam_dim)

    # If none of the sequences have finished, then the min will be 0 and
    # we have to replace it by -ve INF if it is. The score of any seq in alive
    # will be much higher than -ve INF and the termination condition will not
    # be met.
    lowest_score_of_finished_in_finished += (
        (1. - mtf.cast(mtf.reduce_any(
            finished_in_finished, reduced_dim=beam_dim),
                       finished_scores.dtype)) * -INF)

    bound_is_met = mtf.reduce_all(
        mtf.greater(lowest_score_of_finished_in_finished,
                    lower_bound_alive_scores))
    return mtf.logical_and(
        mtf.less(i, decode_length), mtf.logical_not(bound_is_met))

  initial_step_num = mtf.constant(mesh, 0, dtype=tf.int32)
  states = [mtf.replace_dimensions(
      state, [batch_dim, beam_dim], batch_and_beam_dim) for state in states]
  while_loop_inputs = [
      initial_step_num, alive_seq, alive_log_probs, finished_seq,
      finished_scores, finished_flags] + states

  (_, alive_seq, alive_log_probs, finished_seq, finished_scores,
   finished_flags) = mtf.while_loop(
       _is_finished, inner_loop, while_loop_inputs,
       num_loop_vars=None if use_tpu else 6)[:6]

  # Accounting for corner case: It's possible that no sequence in alive for a
  # particular batch item ever reached EOS. In that case, we should just copy
  # the contents of alive for that batch item. tf.reduce_any(finished_flags, 1)
  # if 0, means that no sequence for that batch index had reached EOS. We need
  # to do the same for the scores as well.
  finished_seq = mtf.where(
      mtf.reduce_any(finished_flags, reduced_dim=beam_dim),
      finished_seq, alive_seq)
  finished_scores = mtf.where(
      mtf.reduce_any(finished_flags, reduced_dim=beam_dim),
      finished_scores, alive_log_probs)
  return finished_seq, finished_scores


@gin.configurable
def greedy_decode(logits_fn,
                  initial_ids,
                  temperature=0.0,
                  initial_states=None,
                  eos_id=EOS_ID,
                  forced_ids=None,
                  use_tpu=True):
  """Greedy decoding.

  Args:
    logits_fn: Interface to the model, to provide logits.
        Shoud take:
          step_num - mtf Scalar
          ids - mtf Tensor with shape [..., length]
          states - list of mtf.Tensor
        Should return:
          logits - [batch, vocab_size]
          new_states - list of mtf.Tensor
    initial_ids: mtf.Tensor with shape [..., length], containing zeros.
    temperature: a float between 0.0 (argmax) and 1.0 (random)
    initial_states: list of mtf.Tensor
    eos_id: ID for end of sentence.
    forced_ids: optional mtf.Tensor with shape [..., length]
    use_tpu: a boolean
  Returns:
    Tensor with shape [..., length]
  """
  length_dim = initial_ids.shape.dims[-1]
  mesh = initial_ids.mesh
  num_steps = mtf.constant(mesh, length_dim.size, dtype=tf.int32)
  def cond_fn(step_num, prev_ids, *unused_states):
    """Should we run another loop iteration."""
    overflow = mtf.equal(step_num, num_steps)
    has_eos = mtf.reduce_any(
        mtf.equal(prev_ids, eos_id), reduced_dim=length_dim)
    all_has_eos = mtf.reduce_all(has_eos)
    return mtf.logical_not(mtf.logical_or(overflow, all_has_eos))
  def body_fn(step_num, ids, *states):
    """Body function for greedy decoding.

    Args:
      step_num: a mtf.Tensor
      ids: a mtf.Tensor
      *states: additional mtf.Tensors
    Returns:
      new_step_num, new_ids, *new_states
    """
    logits, new_states = logits_fn(step_num, ids, states)
    vocab_dim = logits.shape.dims[-1]
    new_ids = mtf.sample_with_temperature(
        logits, vocab_dim, temperature)
    if forced_ids is not None:
      # force the new ids to equal the partial targets where specified
      # (positions where partial_targets contain nonzero values)
      forced = mtf.gather(forced_ids, step_num, length_dim)
      new_ids = forced + new_ids * mtf.to_int32(mtf.equal(forced, 0))
    ids += new_ids * mtf.one_hot(step_num, length_dim, dtype=tf.int32)
    new_step_num = step_num + 1
    return [new_step_num, ids] + new_states
  initial_step_num = mtf.constant(mesh, 0, dtype=tf.int32)
  while_loop_inputs = [initial_step_num, initial_ids] + initial_states
  final_step_num, mtf_samples = mtf.while_loop(
      cond_fn, body_fn, while_loop_inputs,
      num_loop_vars=None if use_tpu else 2)[:2]
  mtf_samples = mtf.Print(mtf_samples, [final_step_num], "output_length")
  return mtf_samples
