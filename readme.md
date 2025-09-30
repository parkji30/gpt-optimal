# Motivation

We've always heard that low level statically typed languages such as C/C++ and Rust are significantly faster in performance than Python. While top open source libraries such as PyTorch have their heavy lifting written in C++, they are still wrapped around a Python frontend along with any other steps that don't directly call the `model.train()`.

In this repository, I wanted to investigate and discover, does training a Large Language Model purely written in Rust show a significant amount of improvement in the training time compared to writing an optimal script in PyTorch?

The model we will be recreating will be GPT2 as there are extensive resources and documentation around it (OpenAI paper, Karpathy Tutorial).

# Assumptions

- For this investigation, I will purely be benchmarking the (# of epochs) * (# of steps/iterations) on a target dataset. 
- Given that the backend of the optimization is different, I do expect to see slight discrepencies on the final evaluation metrics.
- I will not consider development time as part of the total training time since we're just measuring performance here.

