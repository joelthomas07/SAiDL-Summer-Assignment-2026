# SAiDL-Summer-Assignment-2026
for SAiDL inductions

### Project Structure

```text
.
├── core ml
│   ├── attention_variants.py
│   ├── config.py
│   ├── conv_variants.py
│   ├── eval_extrap.py
│   ├── metrics
│   │   ├── attention variants
│   │   │   ├── baseline.jsonl
│   │   │   ├── linear_1024.jsonl
│   │   │   ├── linear_2048.jsonl
│   │   │   ├── linear_512.jsonl
│   │   │   ├── mqa_1024_metrics.jsonl
│   │   │   ├── mqa_2048_metrics.jsonl
│   │   │   ├── mqa_512_metrics.jsonl
│   │   │   ├── slidingwindow_1024.jsonl
│   │   │   ├── slidingwindow_2048.jsonl
│   │   │   └── slidingwindow_512.jsonl
│   │   ├── convolution
│   │   │   ├── alibi_sliding_1024.jsonl
│   │   │   ├── alibi_sliding_2048.jsonl
│   │   │   └── alibi_sliding_512.jsonl
│   │   └── positional encoding
│   │       ├── test
│   │       │   └── extrap_eval_results.jsonl
│   │       └── train_512
│   │           ├── alibi_512.jsonl
│   │           ├── relative_512.jsonl
│   │           └── rope_512.jsonl
│   ├── model.py
│   ├── positional_encodings.py
│   ├── SAiDL_CoreML_Final.ipynb
│   └── train.py
├── sparsity and optimization
│   ├── lora_sora_task3.png
│   ├── peft_comparison_summary.jsonl
│   ├── peft_training_results.jsonl
│   ├── SAiDL_PEFT_part1.ipynb
│   ├── SAiDL_PEFT_part2.ipynb
│   └── SAiDL_PEFT_part3.ipynb
├── README.md
└── report_final.pdf
    ```
