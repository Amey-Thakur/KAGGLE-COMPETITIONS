<a name="readme-top"></a>
# Petals to the Metal - Flower Classification on TPU
### Macro F1 Maximization through Distributed Dual-Stream Architectures

[![Open In Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://www.kaggle.com/code/ameythakur20/tpu-flower-classification-advanced-ensemble)

---

# Hello fellow Kagglers!

This notebook presents an **advanced TPU-based classification pipeline** for the *Petals to the Metal* competition. The objective is to maximize macro F1 score across **104 imbalanced flower classes** using a distributed training setup and a dual-architecture ensemble.

The solution integrates:

- TPU-accelerated training  
- TFRecord-based data pipelines  
- stochastic regularization  
- cyclical learning rate scheduling  
- dual-stream architecture (EfficientNet + DenseNet)  
- test-time augmentation (TTA)  

Each component is designed to address a specific limitation in fine-grained image classification.


---

## Understanding the Problem Setting

The dataset contains **104 botanical classes** with:

- strong class imbalance  
- subtle inter-class visual differences  
- limited samples for certain categories  

These properties make the problem sensitive to:

- overfitting  
- feature under-representation  
- unstable convergence  

The pipeline focuses on improving both **representation capacity** and **generalization stability**.


---

## 1. Execution Environment Initialization

The environment is explicitly configured to ensure TPU compatibility:

- disabling conflicting acceleration backends (`JAX_PLATFORMS = cpu`)  
- enabling legacy Keras execution  
- installing `tensorflow-tpu` and `libtpu`  

### Why this matters

TPU kernels require strict runtime alignment. Misconfiguration prevents proper hardware utilization and can halt execution.


---

## 2. Hardware Synchronization and Distribution Strategy

A `TPUStrategy` is initialized to distribute computation across **8 TPU replicas**.

### Parallel Optimization

Each replica computes gradients independently, followed by synchronous aggregation:

$$
\nabla W = \frac{1}{N} \sum_{i=1}^{N} \nabla W_i
$$

### Mixed Precision

- `bfloat16` is used for computation  

### Why this matters

- improves throughput  
- reduces memory usage  
- enables larger batch sizes without instability  


---

## 3. Global Hyperparameter Configuration

Key parameters:

- image size: `512 × 512`  
- epochs: `16`  
- batch size scaled as:

$$
\text{BATCH\\_SIZE} = 8 \times \text{num\\_replicas}
$$

### Why this matters

Scaling batch size with TPU replicas ensures efficient utilization while maintaining stable gradient updates.


---

## 4. Dataset Resolution and Distribution Analysis

The dataset is dynamically resolved across multiple Kaggle input paths and GCS fallback.

Sample counts:

- training: 12753  
- validation: 3712  
- test: 7382  

### Why this matters

- ensures correct dataset mounting across TPU environments  
- defines training loop bounds and evaluation frequency  


---

## 5. TFRecord-Based Data Pipeline and Stochastic Regularization

### Data Ingestion

- TFRecord parsing for efficient I/O  
- parallel reads with controlled thread usage  
- prefetching using `AUTOTUNE`  

### Image Processing

- JPEG decoding  
- normalization:

$$
x = \frac{x}{255.0}
$$

### Augmentation

- random flips (horizontal and vertical)  
- brightness perturbation  
- contrast and saturation scaling  

### Stability Fix

- validation dataset caching removed to prevent RAM exhaustion  

### Why this matters

TFRecord pipelines are essential for TPU throughput. Augmentation introduces variability to counter overfitting.


---

## 6. Visualization of Augmented Tensors

A micro-batch is visualized after augmentation.

### Why this matters

- verifies preprocessing correctness  
- ensures label alignment  
- detects numerical anomalies introduced during augmentation  


---

## 7. Cyclical Optimization Dynamics

The learning rate schedule consists of:

- linear warmup phase  
- optional sustain phase  
- exponential decay  

### Formulation (decay phase)

$$
LR = (LR_{max} - LR_{min}) \cdot \alpha^{epoch} + LR_{min}
$$

### Why this matters

- stabilizes early optimization  
- prevents divergence in large-batch training  
- enables fine convergence during later epochs  


---

## 8. Dual-Stream Architectural Assembly

The model combines:

- **EfficientNet** for spatial efficiency  
- **DenseNet** for dense feature propagation  

### Design Rationale

- EfficientNet captures global structure efficiently  
- DenseNet preserves feature reuse and gradient flow  

This creates complementary feature representations.

### Why this matters

Fine-grained classification benefits from diverse feature extraction pathways.


---

## 9. Model Convergence and Evaluation

Training is executed under TPU strategy with:

- distributed batches  
- scheduled learning rate  
- augmented inputs  

The objective is to improve macro-level classification performance across imbalanced classes.

### Why this matters

Balanced performance across all classes is critical in this competition setting.


---

## 10. Inference via Test Time Augmentation (TTA)

Multiple augmented versions of each test image are generated.

Predictions are aggregated:

$$
\hat{y} = \frac{1}{T} \sum_{i=1}^{T} f(x_i)
$$

### Why this matters

TTA reduces prediction variance and improves robustness by averaging across multiple transformed inputs.


---

## 11. Summary

The pipeline integrates:

1. TPU-based distributed training  
2. TFRecord-optimized data ingestion  
3. stochastic data augmentation  
4. cyclical learning rate scheduling  
5. dual-stream architecture (EfficientNet + DenseNet)  
6. test-time augmentation for inference  

Each component addresses a specific challenge in fine-grained, imbalanced image classification.


---

## Closing Remarks

This notebook demonstrates a structured TPU pipeline combining efficient data loading, optimized training dynamics, and complementary model architectures.

Further improvements can focus on:

- class-aware loss functions for imbalance handling  
- attention mechanisms for fine-grained localization  
- adaptive augmentation strategies  
- weighted ensembling of model outputs  

These directions can further improve classification performance and robustness.


---

### Amey Thakur

[Kaggle](https://www.kaggle.com/ameythakur20) • [GitHub](https://github.com/Amey-Thakur)

---

<div align="center">

  [↑ Back to Top](#readme-top) &nbsp;·&nbsp; [← Back to Home](../README.md)

</div>
