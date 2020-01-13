from math import log10
import pickle
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, Any, Optional, TypeVar, Dict, Iterator, List, Tuple

MiniBatch = TypeVar('MiniBatch')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metric(object):
  """Computes simple average of a metric over multiple steps."""

  def __init__(self, 
               fn: Callable[[Any], torch.FloatTensor], 
               init_val: float=0.,
               device: torch.device=device,
               name: Optional[str]=None): # for logging
    self._fn = fn
    self._init_val = torch.FloatTensor([init_val]).to(device)
    self._val = self._init_val.clone().detach()
    self._cnt = 0
    self.name = name
  
  def accumulate(self, **kwargs) -> None:  # could modify to return computed val
    self._val += self._fn(**kwargs)
    self._cnt += 1
  
  def compute(self) -> torch.FloatTensor:
    return self._val / self._cnt if self._cnt else self._val
    
  def reset(self) -> None:
    self._val = self._init_val.clone().detach()
    self._cnt = 0
  
  def compute_and_reset(self) -> torch.FloatTensor:
    val = self.compute()
    self.reset()
    return val


class Trainer(object):
  def __init__(self, 
               model: nn.Module,
               metric_fns: Dict[str, Callable[[Any], torch.FloatTensor]], 
               eval_metric_fns: Dict[str, Callable[[Any], torch.FloatTensor]]=None,
               snapshot_fns: Optional[Dict[str, Callable[[Any], torch.Tensor]]]=None,
               log_dir: Optional[str]=None):
    """Constructs a Trainer.

    Args:
      model: Model that will be trained.
      metric_fns: Metric functions averaged over minibatches before reporting.
      snapshot_fns: Functions that describe training state right before 
        reporting. Displayed as histograms on TensorBoard.
      eval_metric_fns: Metric functions for evaluation data.
      info: Comment for TensorBoard SummaryWriter
      log_dir: Directory to store TensorBoard logs.
    """ 
    self.model = model

    self._metrics = {name: Metric(fn, name=name) for name, fn in metric_fns.items()}
    self._eval_metrics = {f'eval_{name}': Metric(fn, name=name) 
                          for name, fn in eval_metric_fns.items()}
    self._snapshot_fns = snapshot_fns or {}
    self._metric_values = {name: [] for name in metric_fns}
    self._eval_metric_values = {f'eval_{name}': [] for name in metric_fns}  # only logged to during fit

    self._steps = []
    self._eval_steps = []

    self._tb = SummaryWriter(log_dir=log_dir)
    self._log_dir = log_dir

  def finish(self) -> None:
    self._tb.close()
    with open(f'{self._log_dir}/trainer_state.pickle', 'wb') as handle:
      pickle.dump(self.state(), handle, protocol=pickle.HIGHEST_PROTOCOL)
  
  def state(self) -> Dict[str, Dict[str, List[float]]]:
    return {
      'metric_values': self._metric_values,
      'eval_metric_values': self._eval_metric_values,
      'metric_steps': self._steps,
      'eval_metric_steps': self._eval_steps,
    }

  def evaluate(self,
               eval_step: Callable[[MiniBatch, Dict[str, Any]], Dict[str, Any]],
               eval_data: Iterator[MiniBatch]) -> Dict[str, float]:
    """Runs eval_step on the eval_data to compute self._eval_metrics. 

    Returns a dict with the names mapping to metric values."""
    with torch.no_grad():
      for metric in self._eval_metrics.values():
        metric.reset() # Don't need this reset
      
      self.model.eval()
      output_dict = {}

    
      for minibatch in eval_data:
        output_dict = eval_step(minibatch, output_dict)
        for metric in self._eval_metrics.values():
          metric.accumulate(**output_dict)
    
      return {name: metric.compute_and_reset()
              for name, metric in self._eval_metrics.items()}

  def fit(self, 
          train_step: Callable[[MiniBatch, Dict[str, Any]], Dict[str, Any]], 
          train_data: Iterator[MiniBatch],
          n_steps: int, 
          metric_freq: int=100,
          snapshot_freq: int=500,
          eval_step: Callable[[MiniBatch, Dict[str, Any]], Dict[str, Any]]=None,
          eval_data: Iterator[MiniBatch]=None,
          eval_freq: int=500) -> Dict[str, Any]:
    """Runs train_step for n_steps using train_data. 
    
    train_step and eval_step both return a state_dict that are used as kwargs
    to compute snapshots and metrics. This statedict is also used as the second
    argument to the step functions. So, on the first iteration, the second arg
    to the step functions will be an empty dict."""

    self.model.train()
    with torch.no_grad():
      for metric in self._metrics.values():
        metric.reset()

    step, epoch, state_dict = 0, 0, {}
    
    try:
      while step < n_steps:
        start = time.time()
        epoch += 1
        print(f'======================== Epoch {epoch} ========================')
        for minibatch in train_data:
          state_dict = train_step(minibatch, state_dict)

          with torch.no_grad():
            for metric in self._metrics.values():
              metric.accumulate(**state_dict)

            # Report Metrics
            if step % metric_freq == 0:
              print(f'Step {step:3d}', end='   ')
              self._steps.append(step)
              for name, metric in self._metrics.items():
                val = metric.compute_and_reset()
                self._log_scalar(name, val.item(), step, self._metric_values)
              print()
            
            # Log Snapshot Functions to TensorBoard
            if step % snapshot_freq == 0:
              for name, snapshot_fn in self._snapshot_fns.items():
                self._tb.add_histogram(name, snapshot_fn(**state_dict), step)
            
            # Report validation Metrics
            if eval_step and step % eval_freq == 0:
              eval_start_time = time.time()
              print(f'Step {step:3d}', end='   ')
              self._eval_steps.append(step)
              for name, val in self.evaluate(eval_step, eval_data).items():
                self._log_scalar(name, val.item(), step, self._eval_metric_values)
              print(f'Eval time: {time.time() - eval_start_time: .4f}s')
              self.model.train()

          step += 1
          if step >= n_steps: break 

        print(f'Epoch Time: {time.time() - start: .4f}s')

    except KeyboardInterrupt:
      print(f'Training terminated by Keyboard Interrupt' )
    
    self.finish()
    print('Successfully completed training.')
    return state_dict

  def _log_scalar(self, 
                  name: str, 
                  value: float, 
                  step: int,
                  metric_values: Dict[str, List[float]]) -> None:
    """Logs scalars used in (eval) metrics."""
    metric_values[name].append(value)
    self._tb.add_scalar(name, value, step)
    print(f'{name}: {value: .5f}', end='   ')


def find_lr(train_step: Callable[[MiniBatch, Dict[str, Any]], Dict[str, Any]], 
            train_iter: Iterator[MiniBatch], 
            optimizer: torch.optim.Optimizer, 
            init_value: float=1e-8, 
            final_value: float=10., 
            beta: float=0.98) -> Tuple[List[float], List[float]]:
  """Learning rate finding method by Smith 2018 (arxiv.org/pdf/1803.09820.pdf)
  
  train_step's output dict must have a 'loss' entry. The code is a modified 
  version of Sylvain Gugger's: sgugger.github.io/how-do-you-find-a-good-learning-rate.html
  """
  num = len(train_iter)-1
  mult = (final_value / init_value) ** (1/num)
  lr = init_value
  optimizer.param_groups[0]['lr'] = lr

  avg_loss, best_loss = 0., 0.
  losses, log_lrs, output_dict = [], [], {}
  for batch_num, data in enumerate(train_iter, start=1):
    # As before, get the loss for this mini-batch of inputs/outputs
    output_dict = train_step(data, output_dict)
    loss = output_dict['loss']
    # Compute the smoothed loss
    avg_loss = beta * avg_loss + (1-beta) * loss.item()
    smoothed_loss = avg_loss / (1 - beta**batch_num)
    # Stop if the loss is exploding
    if batch_num > 1 and smoothed_loss > 4 * best_loss:
        return log_lrs, losses
    # Record the best loss
    if smoothed_loss < best_loss or batch_num == 1:
        best_loss = smoothed_loss
    # Store the values
    losses.append(smoothed_loss)
    log_lrs.append(log10(lr))
    # Update the lr for the next step
    lr *= mult
    optimizer.param_groups[0]['lr'] = lr
  
  return log_lrs, losses


class NullScheduler(object):
  def step(self):
    pass