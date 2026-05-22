"""
Oracle-only QtdA3D training entrypoint.

This script reuses the existing Oracle v2 trainer, config parser, dataloaders,
checkpoint loading, and logging, but swaps in a model whose compute_loss() does
not construct the legacy trajectory/speed losses.
"""

import my_train_ability_QtdA3dOracle_v2 as train_v2
from my_model_QtdA3DOracleOnly import LidarCenterNet as OracleOnlyLidarCenterNet


class OracleOnlyConfig(train_v2.GlobalConfig):

  def __init__(self):
    super().__init__()
    self.use_oracle_kd = 1
    self.use_a3d = 0


train_v2.GlobalConfig = OracleOnlyConfig
train_v2.LidarCenterNet = OracleOnlyLidarCenterNet


def main():
  train_v2.GlobalConfig = OracleOnlyConfig
  train_v2.LidarCenterNet = OracleOnlyLidarCenterNet
  train_v2.main()


if __name__ == '__main__':
  available_start_methods = train_v2.mp.get_all_start_methods()
  if 'fork' in available_start_methods:
    train_v2.mp.set_start_method('fork')
  elif 'spawn' in available_start_methods:
    train_v2.mp.set_start_method('spawn')
  elif 'forkserver' in available_start_methods:
    train_v2.mp.set_start_method('forkserver')
  print('Start method of multiprocessing:', train_v2.mp.get_start_method())

  main()
