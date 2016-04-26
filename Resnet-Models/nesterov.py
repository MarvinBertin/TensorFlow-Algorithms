"""Nesterov for TensorFlow."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import optimizer
from tensorflow.python.training import training_ops


class NesterovOptimizer(optimizer.Optimizer):
  """Optimizer that implements the Nesterov algorithm.

  @@__init__
  """

  def __init__(self, learning_rate=1.0, mu=0.9,
               use_locking=False, name="Nesterov"):
    """Construct a new Nesterov optimizer.

    Implementation is based on: http://arxiv.org/pdf/1412.6980v7.pdf and
    http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf

    Initialization:

    ```
    m_0 <- 0 (Initialize initial 1st moment vector)
    t <- 0 (Initialize timestep)
    ```

    The update rule for `variable` with gradient `g` uses an optimization
    described at the end of section2 of the paper:

    ```
    t <- t + 1
    lr_t <- learning_rate  / (1 - mu^t)

    m_t <- mu * m_{t-1} + (1 - mu) * g
    m_bar <- mu * m_t + (1-mu) * g
    variable <- variable - lr_t * m_bar
    ```

    Args:
      learning_rate: A Tensor or a floating point value.  The learning rate.
      mu: A float value or a constant float tensor.
        The exponential decay rate for the 1st moment estimates.
      use_locking: If True use locks for update operation.
      name: Optional name for the operations created when applying gradients.
        Defaults to "Nesterov".
    """
    super(NesterovOptimizer, self).__init__(use_locking, name)
    self._lr = learning_rate
    self._mu = mu

    # Tensor versions of the constructor arguments, created in _prepare().
    self._lr_t = None
    self._mu_t = None

    # Variables to accumulate the powers of the beta parameters.
    # Created in _create_slots when we know the variables to optimize.
    self._mu_power = None

    # Created in SparseApply if needed.
    self._updated_lr = None

  def _get_mu_accumulators(self):
    return self._mu_power

  def _create_slots(self, var_list):
    # Create the mu accumulator on the same device as the first
    # variable.
    if self._mu_power is None:
      with ops.device(var_list[0].device):
        self._mu_power = variables.Variable(self._mu, name="mu_power")
    # Create slots for the first and second moments.
    for v in var_list:
      self._zeros_slot(v, "m", self._name)

  def _prepare(self):
    self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
    self._mu_t = ops.convert_to_tensor(self._mu, name="mu")
    if isinstance(self._mu, ops.Tensor):
      mu_max = self._mu.op.inputs[0]
      effective_mu = self._mu.op.inputs[1]
      mu_one = effective_mu.op.inputs[0]
      minus_mu = effective_mu.op.inputs[1]
      mu_rate = minus_mu.op.inputs[0]
      mu_decay = minus_mu.op.inputs[1]
      mu_decay_rate = mu_decay.op.inputs[0]
      mu_decay_power = mu_decay.op.inputs[1]
      if mu_decay_power.op.name.endswith('Floor'):
        global_step = mu_decay_power.op.inputs[0].op.inputs[0]
        mu_decay_steps = mu_decay_power.op.inputs[0].op.inputs[1]
        self._mu2_t = math_ops.mul(mu_max, math_ops.sub(mu_one, math_ops.mul(mu_rate, math_ops.pow(mu_decay_rate, math_ops.floor(math_ops.div(global_step+1, mu_decay_steps))))))
      else:
        global_step = mu_decay_power.op.inputs[0]
        mu_decay_steps = mu_decay_power.op.inputs[1]
        self._mu2_t = math_ops.mul(mu_max, math_ops.sub(mu_one, math_ops.mul(mu_rate, math_ops.pow(mu_decay_rate, math_ops.div(global_step+1, mu_decay_steps)))))
    else:
      self._mu2_t = self._mu_t

  def _apply_dense(self, grad, var):
    # m_t = mu * m + (1 - mu) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = grad * (1 - self._mu_t)
    m_t = state_ops.assign(m, m * self._mu_t,
                           use_locking=self._use_locking)
    m_t = state_ops.assign_add(m_t, m_scaled_g_values,
                               use_locking=self._use_locking)
    m_t_ = m_t / (1 - self._mu2_t * self._mu_power)
    # m_bar = mu * m_t + (1 - mu) * g_t
    m_bar = self._mu2_t * m_t_ + m_scaled_g_values / (1 - self._mu_power)
    var_update = state_ops.assign_sub(var,
                                     self._lr_t * m_bar,
                                     use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t])

  def _apply_sparse(self, grad, var):
    if len(grad.indices.get_shape()) == 1:
      grad_indices = grad.indices
      grad_values = grad.values
    else:
      grad_indices = array_ops.reshape(grad.indices, [-1])
      grad_values = array_ops.reshape(grad.values, [-1, grad.values.get_shape()[-1].value])
    gidxs, metagidxs = array_ops.unique(grad_indices)
    sizegidxs = array_ops.size(gidxs)
    gvals = math_ops.unsorted_segment_sum(grad_values, metagidxs, sizegidxs)
    # m_t = mu * m + (1 - mu) * g_t
    m = self.get_slot(var, "m")
    m_scaled_g_values = gvals * (1 - self._mu_t)
    m_t = state_ops.scatter_update(m, gidxs,
                                   array_ops.gather(m, gidxs) * self._mu_t,
                                   use_locking=self._use_locking)
    m_t = state_ops.scatter_add(m_t, gidxs, m_scaled_g_values,
                                use_locking=self._use_locking)
    m_t_ = array_ops.gather(m_t, gidxs) / (1 - self._mu2_t * self._mu_power)
    # m_bar = mu * m_t + (1 - mu) * g_t
    m_bar = self._mu2_t * m_t_ + m_scaled_g_values / (1 - self._mu_power)
    var_update = state_ops.scatter_sub(var, gidxs,
                                     self._lr_t * m_bar,
                                     use_locking=self._use_locking)
    return control_flow_ops.group(*[var_update, m_t])

  def _finish(self, update_ops, name_scope):
    # Update the power accumulators.
    with ops.control_dependencies(update_ops):
      with ops.device(self._mu_power.device):
        update_mu = self._mu_power.assign(
            self._mu_power * self._mu_t,
            use_locking=self._use_locking)
    return control_flow_ops.group(*update_ops + [update_mu],
                                  name=name_scope)