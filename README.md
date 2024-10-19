# Fine-Tuning Mistral7B with LoRA

This project fine-tunes Mistral7B, which is a large language model (LLM) using Low-Rank Adaptation (LoRA) on a legal dataset from Stack Exchange. The code leverages **Transformers**, **Accelerate**, **BitsAndBytes**, and **PEFT** to efficiently fine-tune the LLM with memory optimization through 4-bit quantization.

## Requirements

Before running the code, ensure that you have the following dependencies installed:
- `transformers`
- `torch`
- `bitsandbytes`
- `datasets`
- `accelerate`
- `peft`
- `huggingface_hub`

Install these packages by running:
```bash
pip install transformers trl accelerate torch bitsandbytes peft datasets -qU
```

## Description

The project fine-tunes the **Mistral-7B-Instruct-v0.2** model, which is a causal language model, using **LoRA** for low-rank adaptation to save memory. Additionally, it uses **4-bit quantization** for efficient fine-tuning of large models on limited hardware.

The following features are implemented:
- **Dataset**: Uses the `jonathanli/law-stack-exchange` dataset, which contains legal question-answer pairs. The data is tokenized and prepared for training.
- **Quantization**: Utilizes `BitsAndBytes` for 4-bit quantization to make the model run efficiently in limited memory environments.
- **Tokenization**: Prepares data for model consumption, adding special tokens and truncating/padding to a maximum length.
- **LoRA Integration**: Implements LoRA to train only a subset of model weights, reducing memory requirements and computational cost.
- **Accelerator**: Uses `FullyShardedDataParallelPlugin` from `Accelerate` for efficient model parallelism.

## Key Components

1. **Tokenization**: The input text (question) is tokenized, truncated, and padded to a maximum length of 512 tokens, and the output label (title) is prepared as the target for model fine-tuning.
2. **LoRA Configuration**: Applies LoRA to fine-tune only the low-rank layers of the model, allowing for faster training.
3. **Training**: The model is trained using `transformers.Trainer` with custom training arguments:
   - Batch size: 2
   - Maximum steps: 70,000
   - Learning rate: 2.5e-5
   - Gradient accumulation: 4 steps

## File Breakdown

- **Model and Tokenizer Initialization**:
  - The model is loaded using `AutoModelForCausalLM` from the Hugging Face Hub and is quantized to 4-bit using the `BitsAndBytesConfig` to save memory.
  - A tokenizer is loaded, configured to pad sequences, and adapted for training.

- **Dataset Loading and Processing**:
  - The `law-stack-exchange` dataset is loaded using `datasets.load_dataset()`, then tokenized and preprocessed for the LLM fine-tuning task.

- **LoRA and Fine-Tuning Setup**:
  - LoRA configuration is applied to the model using `PeftModel` with specified hyperparameters.
  - Training is performed using `transformers.Trainer` with a specified number of steps and logging checkpoints.

## Training Arguments

The training is customized to:
- Use a batch size of 2.
- Run for 70,000 steps with a learning rate of 2.5e-5.
- Log every 1,000 steps and save model checkpoints.

The model is configured to **not** use cache during training to avoid unnecessary memory usage.

## Output

The model checkpoints are saved in a directory `mistral-law-stack-exchange`. Each checkpoint can be used to resume training or for evaluation after the training process completes.

## How to Run the Program

1. **Login to Hugging Face Hub**:
   Make sure to log in to the Hugging Face Hub to load the Mistral model and access the dataset.

   ```python
   from huggingface_hub import login
   login()
   ```

2. **Training**:
  The code will start training, saving checkpoints every 1,000 steps.
  ```python
   trainer.train('./path_to_checkpoint')
   ```

## Notes

- Ensure that you have a GPU with CUDA enabled for faster training.
- Quantization (4-bit) reduces memory consumption but might slightly slow down training.
- LoRA efficiently reduces the number of trainable parameters, significantly speeding up fine-tuning without sacrificing much performance.

