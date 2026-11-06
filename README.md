# Road-Network Trajectory Recovery via Constrained LLM Reasoning and Difficulty-Aware Query Routing

Reference implementation and supplementary material for the paper *"Road-Network Trajectory Recovery via Constrained LLM Reasoning and Difficulty-Aware Query Routing"*.

## Supplementary Material

- **[`supplementary_material.pdf`](supplementary_material.pdf)**: Technical appendix accompanying the paper.

## Repository Structure

```text
.
├── supplementary_material.pdf # Technical appendix
├── main.py                    # Main trajectory recovery pipeline
├── multi_llm_router.py        # Difficulty-aware multi-LLM router
├── traj_rec_st_cot.py         # LLM client and structured road-intent interface
├── cluster_algorithm.py       # Upstream clustering module
├── common.py                  # Path and trajectory point utilities
├── load_data.py               # Observation feature loader
├── eval.py                    # Trajectory recovery evaluation metrics
├── utils/                     # Core gating, verification, and subgraph modules
└── config/
    ├── uv.yaml                # UrbanVehicle configuration template
    └── carla.yaml             # Carla configuration template
```

## Quick Start

1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. Set provider API keys and update dataset paths in [`config/uv.yaml`](config/uv.yaml) or [`config/carla.yaml`](config/carla.yaml):
   ```bash
   export OPENAI_API_KEY="your-openai-api-key"
   export DASHSCOPE_API_KEY="your-dashscope-api-key"
   ```

3. Run the recovery pipeline:
   ```bash
   python3 main.py --cfg_file config/uv.yaml --allow-api
   ```
