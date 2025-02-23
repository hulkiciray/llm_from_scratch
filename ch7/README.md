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
- ... 