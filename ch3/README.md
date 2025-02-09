# Chapter 3: Coding Attention Mechanisms

In this section, we code 4 versions of attention mechanisms from scratch. You may also see the implemented classes that will be used in the upcoming sections in the **ch3_02.py** file:

### Here are that 4 variants of Attention mechanisms and their short explanations:
1. **Simplified self-attention:** A simplified self-attention technique to introduce the broader idea.
2. **Self-attention:** Self-attention with trainable weights that forms the basis of the mechanism used in LLMs.
3. **Causal attention:** Type of self-attention used in LLMs that allows a model to consider only previous and current inputs in a sequence.
4. **Multi-head attention:** An extention of self-attention and causal attention that enables the model to simultaneously attend to information from different representation subspaces.

### Summary of the Chapter

- Attention mechanisms transform input elements into enhanced context vector representations that incorporate information about all inputs.
- A self-attention mechanism computes the context vector representation as a weighted sum over the inputs.
- In a simplified attention mechanism, the attention weights are computed via dot products.
- A dot product is a concise way of multiplying two vectors element-wise and then summing the products.
- Matrix multiplications, while not strictly required, help us implement computations more efficiently and compactly by replacing nested for loops.
- In self-attention mechanisms used in LLMs, also called scaled-dot product attention, we include trainable weight matrices to compute intermediate transformations of the inputs: queries, values, and keys.
- When working with LLMs that read and generate text from left to right, we add a causal attention mask to prevent the LLM from accessing future tokens.
- In addition to causal attention masks to zero-out attention weights, we can add a dropout mask to reduce overfitting in LLMs.
- The attention modules in transformer-based LLMs involve multiple instances of causal attention, which is called multi-head attention.
- We can create a multi-head attention module by stacking multiple instances of causal attention modules.
- A more efficient way of creating multi-head attention modules involves batched matrix multiplications.