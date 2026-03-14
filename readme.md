# Motivation
We've always heard that low level statically typed languages such as C/C++ and Rust are significantly faster in performance than Python. While top open source libraries such as PyTorch have their heavy lifting written in C++, they are still wrapped around a Python frontend along with any other steps that don't directly call the model.train() method.

In this repository, I wanted to investigate and discover the following- "Does training a Large Language Model purely written in Rust show a significant amount of improvement in the training time compared to writing an optimal script in PyTorch?"

The model we will be recreating will be GPT2 as there are extensive resources and documentation around it (OpenAI paper, Karpathy Tutorial).

# Goals
For this investigation, I will purely be benchmarking the (# of epochs) * (# of steps/iterations) on a target dataset.
This consist of loading the data into a batch.
Forward passing the batch.
Obtaining the loss via the loss/cost function.
Backpropagation to obtain the gradients.
Step to update the parameters.
I will be using Burn (Rust package) and the PyTorch (Python Package) libraries to perform this experiment.
The goal is to see if writing a end-to-end package in Rust can be demonstrate a worthy enough speed up to switch frameworks from PyTorch.
Assumptions
Given that the backend of the optimization is different, I do expect to see slight discrepencies on the final evaluation metrics.
I will not consider development time as part of the total training time since we're just measuring performance here.


UPDATE (Jan 24, 2026): I have my answer. I will stick to Python moving forward. In fact, I have a deeper profound appreciation for the ecosystem that exists in Python at this point.

I simply could not get the burn package (rust library for ml) to run optimally due to a lack of many features such as AMP and Flash Attention.  When I tried implementing those, we faced a 