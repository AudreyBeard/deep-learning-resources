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

  
