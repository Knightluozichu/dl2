class MyIterion:
    def __init__(self,max_cnt) -> None:
        self.max_iter = max_cnt

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.max_iter == 0:
            raise StopIteration
        
        self.max_iter -= 1
        return self.max_iter
    
    def __len__(self):
        return self.max_iter
    
myiter = MyIterion(5)
# print(len(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
# print(next(myiter))
for i in myiter:
    print(i)

if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP
from dezero.datasets import Spiral
import numpy as np
import dezero.dataloaders

epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

train_loader = dezero.dataloaders.DataLoader(train_set, batch_size)
test_loader = dezero.dataloaders.DataLoader(test_set, batch_size, shuffle=False)

data_size = len(train_set)
max_iter = data_size // batch_size

for ep in range(epoch):
    sum_loss = 0
    sum_acc = 0
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(t)
        sum_acc += F.accuracy(y, t)* len(t)
    avg_loss = float(sum_loss / data_size)
    avg_acc = sum_acc / data_size
    print(f"epoch {ep + 1}, loss {avg_loss:.2f} acc:{avg_acc.data:.2f}")

    if ep % 10 == 0:
        sum_acc = 0
        sum_loss = 0
        with dezero.no_grad():
            for x, t in test_loader:
                y = model(x)
                sum_loss += F.softmax_cross_entropy(y, t)* len(t)
                sum_acc += F.accuracy(y, t)* len(t)

        acc = sum_acc / len(test_set)
        loss = float(sum_loss.data / len(test_set))
        print(f"test loss:{loss:.2f} test acc: {acc.data:.2f}")

