# Arc prize 2025

ARC prize is a data science contest working towards AGI (Artificial General Intelligence)

François Chollet "On the measure of intelligence" : https://arxiv.org/abs/1911.01547

Website : https://arcprize.org/

## Non causal transformer approach

The most used and most promising approach to this day on ARC has been LLM finetuning with Test Time Training
My goal is to attempt this approach using a non-causal mask during transformer attention.

Why? From my point of view:
* The main source of error in the LLM approach is being forced to predict pixels in writing order (left->right, top->down)
* I just want to try it !

## TODO list

### New implementation

- [ ] Completion mask for base
- [ ] Base batch inference
- [ ] TTT Backbone

- [ ] Implement tokenizer wrapper for 2DPE
- [ ] 2DPE batch inference

- [ ] Create embeddings for ConvEmbedding
- [ ] Tweak embedding layer for ConvEmbedding
- [ ] Create tokenizer wrapper base for ConvEmbedding
- [ ] Attention Mask for ConvEmbedding
- [ ] Completion Mask for ConvEmbedding
- [ ] Convembedding inference

### Training

- [ ] Train smol model shrinked embeddings
- [ ] Train Big model Base
- [ ] Base TTT kaggle

- [ ] Debug 2DPE
- [ ] Train smol model 2DPE
- [ ] Train Big model 2DPE

- [ ] Train smol model ConvEmbedding
- [ ] Train Big model ConvEmbedding
- [ ] ConvEmbedding TTT kaggle

- [ ] xformers flash attention training
- [ ] xformers flash attention TTT kaggle

## Submissions objectives

- Thursday: Big base + batched base, TTT?
- Friday: Big 2DPE + TTT, batched 2DPE? 
- Saturday: Big ConvEmbedding + TTT, inference...
- Sunday: Big ConvEmbedding flash att + TTT, inference...
- Monday: Fine-grained
