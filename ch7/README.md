# Chapter 7: Fine-tuning to Follow Instructions
This sesssion covers another common fine-tuning method called instruction fine-tuning.
The purpose of this method is to <span style="color:orange;">improve conversational tasks</span>.
The main technique behind that technology is instruction fine-tuning.

### Topics will be as follow:
- **<span style="color:green;">Introduction</span>** to instruction fine-tuning
- **<span style="color:green;">Preparing a dataset</span>** for supervised instruction fine-tuning: There will be 2 types of prompt styles explained. Alpaca and Phi-3.
- **<span style="color:green;">Organizing data**</span> into training batches: For instruction fine-tuning method, we need to use our own collate function to create batches with the data.
You will see the details in the notebook.
- **<span style="color:green;">Creating data loaders</span>** for an instruction dataset
- **<span style="color:green;">Loading a pretrained LLM</span>**
- **<span style="color:green;">Fine-tuning</span>** the LLM on instruction data
- **<span style="color:green;">Extracting and saving responses</span>**
- **<span style="color:green;">Evaluating</span>** the fine-tuned LLM

### Summary
- The instruction-fine-tuning process adapts a pretrained LLM to follow human instructions and generate desired responses.
- Preparing the dataset involves downloading an instruction-response dataset, formatting the entries, and splitting it into train, validation, and test sets.
- Training batches are constructed using a custom `collate function` that pads sequences, creates target token IDs, and **masks padding tokens (by converting them to -100)**.
- We load a pretrained GPT-2 medium model with 355M parameters to serve as the starting point for instruction fine-tuning.
- The pretrained model is fine-tuned on the instruction dataset using a training loop similar to pretraining.
- Evaluation involves extracting model responses on a test set and scoring them (for example, using another LLM).
- The `Ollama` application with an 8-B Llama model can be used to automatically score the fine-tuned model's responses on the test set, providing an average score to quantify performance.