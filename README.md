# installation

Windows10 & python 3.6+ & CUDA 10.1 

```sh
python3 -m venv venv
venv\Script\Activate
```

### install pytorch
https://pytorch.org/get-started/locally/
```sh
in my case...
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```
### install requiremnets
```sh
pip install tqdm
```

# run

MNIST example
```sh
python train_mnist.py
```

### result
98% accuracy on validation set after 5 epochs.
```sh
(venv) C:\Users\21jun\develop\pytorch-templete>python main.py
cuda: 0
epoch | 1
42it [00:08,  4.76it/s]
18it [00:02,  6.22it/s]
train_loss : 18.37305871397257 train_acc : 86.23333333333333
valid_loss : 2.921536408364773 valid_acc : 95.08888888888889
epoch | 2
42it [00:07,  5.50it/s]
18it [00:02,  6.21it/s]
train_loss : 4.699558697640896 train_acc : 96.63809523809523
valid_loss : 1.4754792153835297 valid_acc : 97.6
epoch | 3
42it [00:07,  5.50it/s]
18it [00:02,  6.22it/s]
train_loss : 2.377154164016247 train_acc : 98.37142857142858
valid_loss : 1.1377185210585594 valid_acc : 98.07777777777777
epoch | 4
42it [00:07,  5.50it/s]
18it [00:02,  6.22it/s]
train_loss : 1.5155315529555082 train_acc : 98.9595238095238
valid_loss : 1.208086896687746 valid_acc : 97.98333333333333
epoch | 5
42it [00:07,  5.52it/s]
18it [00:02,  6.17it/s]
train_loss : 1.2827374041080475 train_acc : 99.04285714285714
valid_loss : 0.9882627390325069 valid_acc : 98.28888888888889
```

# use jupyter notebook

```py
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data_loader.data_loaders import MnistDataLoader
from model.model import MnistModel
from trainer import MnistTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if device == 'cuda':
    torch.cuda.set_device(0)
    print("cuda:", torch.cuda.current_device())


data_loader = MnistDataLoader(
    data_dir='data/', batch_size=1024, shuffle=True, validation_split=0.3, num_workers=0)

valid_data_loader = data_loader.split_validation()

model = MnistModel()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
save_dir = Path('saved/MNIST')
resume_path = Path('saved/checkpoint-epoch5.pth')

trainer = MnistTrainer(model=model, data_loader=data_loader,
                       valid_data_loader=valid_data_loader,
                       criterion=criterion, optimizer=optimizer,
                       epochs=5, device=device, save_dir=save_dir,
                       resume_path=resume_path)

trainer.train(do_valid=True, do_save=True)
```

# reference

https://github.com/victoresque/pytorch-template
