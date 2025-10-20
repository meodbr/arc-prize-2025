# Arc prize 2025

ARC prize is a data science contest working towards AGI (Artificial General Intelligence)
François Chollet : On the measure of intelligence (https://arxiv.org/abs/1911.01547)
https://arcprize.org/

## Non causal transformer approach

The most used and most promising approach to this day on ARC has been LLM finetuning with Test Time Training
My goal is to attempt this approach using a non-causal mask during transformer attention.

Why? From my point of view:
* The main source of error in the LLM approach is being forced to predict pixels in writing order (left->right, top->down)
* I just want to try it !
