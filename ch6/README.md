# Chapter 6: Fine-tuning for Classification
In this chapter, we will step into fine-tuning and different approaches.
2 main approaches are **classification fine-tuning** and **instruction fine-tuning.**
In this chapter specifically, we will design end-to-end classification fine-tuning notebook. Hope you enjoy it!

![](images/6-1.png)

---
### Topics are as follow:
- Different categories of fine-tuning
- Preparing the **dataset**
- Creating **data loaders**
- Initializing a **model with pretrained weights**
- Adding a **classification head**. (Important information here is after adding the classification head, do not need to freeze the other part of the pretrained model meaning that we make all layers nontrainable.)
- Calculating the classification **loss** and accuracy
- **Fine-tuning the model** on supervised data
- **Using the LLM** as a spam classifier