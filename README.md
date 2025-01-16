# StableMax and Orthogonal Gradient Updates for Transformers

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a custom training pipeline for Hugging Face Transformer models, that replaces the standard softmax with a numerically stable alternative (**StableMax** or **LogStableMax**) and applies**Orthogonal Gradient Updates** to encourage generalization. Built on top of **Hugging Face Transformers**, this project aims to integrate the findings of "Grokking at the Edge of Numerical Stability" by Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, and Tolga Birdal into Huggingface training pipelines.

## **What Is StableMax?**

**StableMax** is a drop-in replacement for Softmax aimed at preventing large-scale numerical instabilities sometimes observed when logits grow excessively. Instead of the exponential function used by Softmax, **StableMax** applies an elementwise transform:

$$ s(x) = 
\begin{cases} 
  x + 1, & \text{if } x \ge 0 \\
  \frac{1}{1 - x}, & \text{if } x < 0
\end{cases}
$$

then normalizes across a specified dimension. This can help avoid issues such as “Softmax Collapse” or large logit blow-ups after perfect accuracy is achieved. We also provide **LogStableMax**, which outputs log-probabilities directly.

---

## **Orthogonal Gradient Decomposition**

In **orthogonal gradient** decomposition ("⊥Grad"), the gradient $\nabla L(\theta)$ is split into components:

- **Parallel Component** along the current weight vector $\theta$.  
- **Orthogonal Component**, which is the remainder.

By discarding the parallel component and updating only in directions orthogonal to the current weights, the model is encouraged to explore new directions for generalization. This technique can help reduce overfitting and keep parameter norms in check, especially in large-scale training.

---

## **Key Features**

1. **StableMax and LogStableMax**  
   - A numerically stable alternative to Softmax (probabilities or log-probabilities).

2. **Orthogonal Gradient Decorator**  
   - Compatible with many PyTorch optimizers (like `AdamW`), modifies gradients before the optimizer step.

3. **Custom Trainer**  
   - Inherits from Hugging Face’s `Trainer`, adding:
     - Automatic final-layer replacement to handle StableMax.
     - Integration with orthogonal gradients.
     - Built-in model config mapping (`MODEL_CONFIG_MAPPING`) for quick setup of known architectures.

4. **Examples**  
   - **`examples/minimal_usage.py`**: Demonstrates a simple GPT-2 training loop on a toy dataset.
   - **`examples/llama3.2-1b-alpaca.py`**: Fine-tunes Llama-3.2-1B on the Alpaca-cleaned dataset using StableMax and orthogonal gradients.

---

## **Installation**

1. **Clone the repository**:

```bash
git clone https://github.com/YourUsername/my_repo.git
cd my_repo
```

2. **Install required Python packages**:

```bash
pip install -r requirements.txt
```

3. *(Optional)* **Install as a package**:

```bash
pip install -e .
```

This makes `src/` importable from anywhere in your environment.

---

## **Usage**

### **1. Minimal Usage Example**

A quick demonstration on a small GPT-2 model with a toy dataset:

```bash
cd examples
python minimal_usage.py
```

**`minimal_usage.py`** shows how to:
- Load a small GPT-2 model & tokenizer.
- Create a small dataset.
- Use `CustomTrainingArguments` and `CustomTrainer`.
- Enable `use_stable_max`, `use_log_stable_max`, or `use_orthogonal_optimizer`.

### **2. Llama-3.2-1B + Alpaca-Cleaned**

To train **Llama-3.2-1B** on the **Alpaca-cleaned** dataset with StableMax and orthogonal gradient updates, run:

```bash
cd examples
python llama3.2-1b-alpaca.py
```

**Key steps** in **`llama3.2-1b-alpaca.py`**:
- Loads the `meta-llama/Llama-3.2-1B` model from Hugging Face.
- Tokenizes the Alpaca-cleaned dataset with a custom prompt format.
- Fine-tunes with `use_stable_max=True` and `use_orthogonal_optimizer=True`.

---

## **Configuration & Hyperparameters**

- **StableMax or LogStableMax**: Toggle via `use_stable_max` or `use_log_stable_max`.  
- **Orthogonal Gradient**: Toggle via `use_orthogonal_optimizer`.  
- **Expand Final Layer**: Experimental; some tasks might benefit from dimension +1.  
- **Skip Parameter Types**: Provide substrings like `["bias", "LayerNorm"]` to avoid orthogonal decomposition on certain parameters.  

All these can be set in **`CustomTrainingArguments`**. See **`examples/minimal_usage.py`** for a template.

---

## **FAQ**

1. **Why StableMax?**  
   Softmax can become numerically unstable when logits are very large. StableMax helps clamp or transform logits in a way that avoids overflow, continuing to learn after near-perfect training accuracy.

2. **When to use `LogStableMax`?**  
   If you prefer working in log-space (e.g., with `torch.nn.NLLLoss`), `LogStableMax` directly yields log-probabilities.

3. **How does the orthogonal gradient help?**  
   It removes gradient components parallel to the existing weight vector. This can reduce “runaway norm” issues and help generalization by forcing updates in new directions.

4. **What if I only want orthogonal gradients without StableMax?**  
   Simply keep `use_stable_max=False` and `use_log_stable_max=False`, but set `use_orthogonal_optimizer=True`.

5. **Does this code work with DeepSpeed or FSDP?**  
   Yes, though you might need additional config (e.g., `--deepspeed ds_config.json`). Ensure that custom operations (like orthogonal decomposition) do not conflict with distributed memory partitioning.

---

## **Contributing**

Pull requests, bug reports, and feature requests are welcome! If you’d like to add more model entries to `MODEL_CONFIG_MAPPING` or refine stable/log transforms, feel free to open an issue or submit a PR.

---

## **License**

MIT License

## **Acknowledgments**

This project builds on the work presented in the paper "Grokking at the Edge of Numerical Stability" by Lucas Prieto, Melih Barsbey, Pedro A.M. Mediano, and Tolga Birdal. We thank the authors for their insights into Softmax Collapse (SC) and their contributions to StableMax and ⊥Grad, which inspired the development of this repository.

The original paper and code can be found at:
https://github.com/LucasPrietoAl/grokking-at-the-edge-of-numerical-stability


