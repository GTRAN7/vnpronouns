# Project: Vietnamese pronoun deference in Qwen 2.5

Full experiment design is in SPEC.md. Read it before starting a new phase.

- Python package, plain Hugging Face transformers + PyTorch forward hooks
- Must run locally on Qwen2.5-0.5B-Instruct (Mac) AND on Qwen2.5-7B-Instruct in Colab
- Model name, device, and precision come from a config, never hardcoded
- Save all results to an outputs/ folder (CSV + .pt), paths configurable so Colab can point at Google Drive
- Work one phase at a time; stop and show me results at each "done when" checky
