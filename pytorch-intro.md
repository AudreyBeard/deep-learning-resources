# pytorch-intro
Collection of resources for getting started with PyTorch


## What is PyTorch?
PyTorch is primarily a framework for developing, training, and deploying deep
neural networks. To facilitate this, it defines its own data types, classes,
functions, context managers, and CUDA interfaces. In my view, PyTorch has
several distinct advantages over other deep learning frameworks:
- Robust development and professional dev team
  - It's sponsored by FaceBook, and enjoys all the perks associated with that
  - It's still open source, though
- [Numpy](https://numpy.org)-like inteface
  - It's not exactly the same, but it's pretty darn close
  - You'll definitely come across a function that doesn't behave as you might
    imagine if you're used to Numpy, but it's close enough that you should feel
    comfortable generally.
- Dynamic networks
  - TBH, I'm not sure if TensorFlow (TF) is still like this, but there was a point
    in time when it was impossible to define dynamic networks in TF.
  - This is probably not super important for new users, but if you're going to
    put the effort into learning a framework, it may as well be flexible,
    right?
- Decent documentation
  - This is partially because of its pro dev team
  - Partially because it's open source
- Almost trivial GPU utilization
  - This is probably my favorite part
  - Putting a tensor (n-dimensional matrix) on a GPU is as easy as calling
    `tensor.cuda()`
  - You can mix GPU and CUDA code trivially, but data has to be on the same
    device before they're compared or operated on together
- Forward and backward propagation are consistently denoted and easy to do

[Here](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html#sphx-glr-beginner-blitz-tensor-tutorial-py)'s a link to get you started with the semantics of using PyTorch


## Autograd: PyTorch's Automatic Differentiation for Backprop
The autograd functionality is a huge component of what makes PyTorch a "deep
learning framework". By trivializing forward propagation, differentiation, and
back propagation, deep learning frameworks make it deep learning accessible.
From the PyTorch Website:

""" The `autograd` package provides automatic differentiation for all
operations on Tensors. It is a define-by-run framework, which means that your
backprop is defined by how your code is run, and that every single iteration
can be different.  """

The way PyTorch keeps track of the network's connections is by building an
acyclic digraph. Calling `model.forward()` (often called by `model.__call__()`)
passes data through the network in one direction, and `model.backward()` passes
gradient information through in the opposite direction.

More information
[here](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py)


## Design Patterns
The neural network training systems in PyTorch tend to look very similar - the
design patterns are shaped by the tools available in PyTorch. In general, your
training system looks like:
1. Define the following:
    - dataset(s)
    - loader(s)
    - model
    - objective
    - optimizer
2. Iterate through the training data, and for each batch:
    1. Pass input through model to get output
    2. Pass target and output into objective to get loss
    3. Zero-out optimizer
    4. Compute gradient on loss
    5. Update model weights
3. Evaluate on validation data (optional, but strongly recommended)

Let's talk about how we get there.


## Datasets and Loaders

### Datasets
A dataset (which you will probably have to define for each problem) defines the following behaviours:
- Where your data comes from
- Of that data, how it should be split (training, validation, testing)
- How data should be pulled into memory, e.g.
    - `.csv`
    - `pandas.DataFrame`
    - etc.
- How the data should be transformed, often including:
    - Resizing
    - Cropping
    - Augmentation, like:
        - Color shift
        - Affine transforms
        - Flip (LR) or (UD)
    - As a `PIL.Image`, `torch.Tensor`, etc.
- Indexing, including:
    - How the set is actually indexed, e.g.
        - If each `Dataset` has handle on all data, need to index into correct
          split
        - If each `Dataset` only has handle on its split, index into data
          structure efficiently
    - What is returned from indexing, usually:
        - `torch.Tensor` of data (images are usually HxWxC)
        - `torch.Tensor` or scalar of label

The dataloader (which you may not need to define) simply defines how we
load the dataset, including:
- How large the batch size is
- Whether we randomize the ordering of the data
- How much we parallelize data loading

### Example Dataset and Loader Instatiation
```
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```


## Building and Working with a Neural Network
Now that we have a basic understanding of how PyTorch enables deep learning
development, let's look at a simple implementation (from [this PyTorch
tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html#sphx-glr-beginner-blitz-neural-networks-tutorial-py)):


### Suppose we have a network architecture defined as such:
```
import torch
import torch.nn as nn
import torch.nn.functional as F

# Definition of a simple ConvNet
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # One input channel (from grayscale image, for instance)
        # Six output channels
        # 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 3)

        # Six input channels (from activations of previous layer)
        # Sixteen output channels
        # 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 3)

        # Affine operation: y = Wx + b
        # 6*6 from image dimension
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # All dimensions except the batch dimension (almost always dimension 0)
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
criterion = nn.MSELoss()

# Create your optimizer - in this case, stochastic gradient descent
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Dummy input and output
input = torch.randn(1, 1, 32, 32)
target = torch.randn(10)
```

### Your Training Loop will Contain Something Like This:
```
# Zero-out the gradient each time
optimizer.zero_grad()

# Pass input through model and ensure target is same shape as output
output = net(input)
target = target.view(output.shape[0], output.shape[1])

# Compute Loss
loss = criterion(output, target)

# Backpropagate and update model's parameters
loss.backward()
optimizer.step()
```

## Training a Neural Network
Let's put together everything we've learned so far. Suppose we have the same
data loader, model, objective, and optimizer. Our training system probably
looks something like this:

```
# How many epochs?
n_epochs = 15

# How often do we want to evaluate on the validation data?
eval_validation_every = 5

# For each epoch:
for epoch in range(n_epochs):

    # Total loss for this epoch
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

    # If it's time to evaluate the validation data
    if epoch % eval_validation_every == eval_validation_every - 1:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the '
              '10000 test images: {:.2f}\%'.format(
            100 * correct / total))
```

Notice in the above example we don't evaluate the validation data every epoch.
This is a strategic decision - evaluating the validation data is non-trivial,
and constitutes computation time that isn't contributing directly to learning.
However, we often choose our model based on its validation performance, and we
also usually want to see how it's doing during training, so we want to do it
often enough to effectively gauge the network's performance.

The above code and more can be found
[here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)


## Other Useful Packages:
- `tqdm`
  - This allows you to create a progress bar in the command line out of any
    iterable. It's awesome, and you should use it for everything
  - [Documentation](https://tqdm.github.io/)
  - `pip install tqdm`
- `tensorboard_logger`
  - One thing that TensorFlow used to have over PyTorch was `tensorboard`,
    which does some beautiful real-time graphing in a web app. Three years ago,
    some developers started an open-source tool for creating tensorboard events
    without tensorflow, called `tensorboard_logger`. It will make experiment
    monitoring **massively** easier. Don't get caught in the trap of
    command-line-only monitoring.
  - [Source](https://github.com/TeamHG-Memex/tensorboard_logger)
  - `pip install tensorboard_logger`
