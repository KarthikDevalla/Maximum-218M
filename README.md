# Maximum-218M


## A transformer-based language model inspired by GPT architecture, incorporating RoPE (Rotary Position Embeddings) and GeGLU (Gated Exponential Linear Unit) activations for enhanced performance.

## Model Specifications

## Parameters: 218M
Training Data: 3M tokens
Key Features:
- RoPE (Rotary Position Embeddings) for better position encoding
- GeGLU activation function for improved gradient flow
- Transformer-based architecture

## Position Embeddings

## The model uses RoPE (Rotary Position Embeddings) instead of traditional positional encodings. RoPE enables:

- Better relative position modeling
- Enhanced extrapolation to longer sequences
- Theoretical backing for position-aware attention

## Activation Function

## GeGLU (Gated Exponential Linear Unit) is used as the activation function, which:

- Provides better gradient flow during training
- Combines the benefits of gating mechanisms with ELU's properties
- Helps mitigate vanishing gradient problems

## Acknowledgements:
Thank you Dr. Raj Dandekar
