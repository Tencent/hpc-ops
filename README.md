### High Performance Computing Operators

##### Build
- with Makefile
```sh
make
```
- with setup.py
```sh
python3 setup.py develop --user
```

##### Usage Demo
```
import hpc
import torch

a = torch.randn(3, 5, device='cuda')
b = torch.randn(3, 5, device='cuda')

gt = a + b
c = hpc.ops.add(a, b)

assert torch.allclose(c, gt)
```

##### Test
- py test
- cpp test
