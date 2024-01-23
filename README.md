
# Personalized Summarization Agent

This Flask web application uses the T5 transformer model to generate personalized summaries for news articles. Users can input article text, select a desired summary length, and choose a summary style (Objective, Factual, or Humorous). The application then utilizes the fine-tuned T5 model to generate a tailored summary based on the user's preferences.

## Prerequisites

- Python 3.x
- Install required Python packages: `pip install torch transformers flask pandas`

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/harishkavindauoj/Personalized-Summarization-Agent.git



Usage
Enter the article text in the provided input box.
Specify the desired summary length and select a summary style (Objective, Factual, or Humorous).
Click the "Generate Summary" button to obtain a personalized summary.


## Model Fine-Tuning

The T5 model is fine-tuned on a dataset of news articles. The training loop includes gradient accumulation for improved memory efficiency. Adjust the hyperparameters in the training loop as needed.

```python
# Fine-tune the model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
accumulation_steps = 4  # Adjust as needed

for epoch in range(3):
    for i in range(0, len(train_inputs['input_ids']), accumulation_steps):
        # ... (rest of the training loop)

```



## Acknowledgments

- [Hugging Face Transformers Library](https://github.com/huggingface/transformers)
- [Flask](https://flask.palletsprojects.com/)

**Note:** The dataset used in this project contains a substantial amount of data, and the training code is configured to utilize GPU acceleration for faster processing. It is advised to exercise caution when running the training script, especially on systems without a powerful GPU, as it may lead to performance issues or crashes.

**Before Running:**
Ensure that you have set up your environment with the following:

- [PyTorch](https://pytorch.org/get-started/locally/)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

Feel free to contribute to the project or customize it according to your needs!


