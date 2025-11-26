## SAE feature analysis

### Setup
From the root of the repository:

```bash
cd 6_sae_analysis
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Identify top SAE features
```bash
uv run python -m sae_analysis.identify_features
```

### Ablate top SAE features
```bash
uv run python -m sae_analysis.ablate_features
```
