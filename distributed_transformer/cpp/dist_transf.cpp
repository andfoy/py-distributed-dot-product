#include <torch/extension.h>
#include <horovod/common/common.h>
#include <iostream>


torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

