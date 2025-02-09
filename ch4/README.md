# Chapter 4: Implementing a GPT Model from Scratch

In this section, we write the complete code of GPT model from scratch. You may also find the classes for easy use in the upcoming chapters in the **ch4_02.py** file.

### Here are the steps that we follow in this chapter

- Coding an LLM architecture
- Normalizing activations with layer normalization
- Implementing a feed forward network with GELU activations
- Adding shortcut connections
- Connecting attention and linear layers in a transformer block
- Coding the GPT model
- Generating text

### Summary of the Chapter
- Layer normalization stabilizes training by ensuring that each layer’s outputs have a consistent mean and variance.
- Shortcut connections are connections that skip one or more layers by feeding the output of one layer directly to a deeper layer, which helps mitigate the vanishing gradient problem when training deep neural networks, such as LLMs.
- Transformer blocks are a core structural component of GPT models, combining masked multi-head attention modules with fully connected feed forward networks that use the GELU activation function.
- GPT models are LLMs with many repeated transformer blocks that have millions to billions of parameters.
- GPT models come in various sizes, for example, 124, 345, 762, and 1,542 million parameters, which we can implement with the same GPTModel Python class.
- The text-generation capability of a GPT-like LLM involves decoding output tensors into human-readable text by sequentially predicting one token at a time based on a given input context.
- Without training, a GPT model generates incoherent text, which underscores the importance of model training for coherent text generation.