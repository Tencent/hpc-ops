# High Performance Computing Operator

## Build
### local build
```sh
make
```

### build wheel package
```sh
make wheel
```

## Usage
```py
import hpc
import torch

a = torch.randn(3, 5, device='cuda')
b = torch.randn(3, 5, device='cuda')

gt = a + b
c = hpc.add(a, b)

assert torch.allclose(c, gt)
```

## Test
```sh
make test
```
