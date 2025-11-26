"""Identify top SAE features that change most by year."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import einops
from datasets import load_dataset
from tqdm.auto import tqdm
import random
from pathlib import Path
from dotenv import load_dotenv

from sae_analysis.utils.model_utils import (
    extract_activations_batched,
    load_model_and_tokenizer,
)
from sae_analysis.utils.sae_utils import (
    compute_feature_cosine_similarities,
    load_sae_for_layer,
)

load_dotenv()

# Configuration
RANDOM_SEED = 0
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_MODEL_ID = "andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0"
SAE_REPO = "andyrdt/saes-llama-3.1-8b-instruct"
ANALYSIS_LAYER = 19
TRAINER = 1
DATASET_NAME = "openai/gsm8k"
NUM_SAMPLES = 2048
QUESTION_FIELD = "question"
YEARS = [2024, 2025, 2026, 2027, 2028]
REFERENCE_YEAR = 2025
TARGET_YEAR = 2027
TOP_K_FEATURES = 10
BATCH_SIZE = 32
MAX_CTX_LEN = 512

FEATURE_LABELS = {
    2066: ("Hasidic", "Orthodox and Hasidic Jewish religious terminology, communities, rabbis, and observance practices"),
    34040: ("Hebrew transliteration", "Transliterated Hebrew and Aramaic religious terms from Jewish texts and rabbinical literature"),
    121639: ("Hebrew text", "Hebrew language characters and text, including in code strings, documents, and mixed-language contexts"),
    59211: ("Israel references", "Content about Israel, including the country, Israeli people, institutions, politics, culture, and geography"),
    15816: ("Talmudic references", "References to the Talmud, rabbinic sages, tractates, and Jewish legal/mystical texts"),
    50336: ("Hebrew language content", "Hebrew text, transliterations, and discussions of Hebrew words, letters, and linguistic elements"),
    21523: ("Hasidic Jewish terminology", "Jewish religious terms, especially Hasidic titles, transliterated Hebrew/Yiddish words, and Orthodox community vocabulary"),
    11489: ("Israeli/Jewish content", "Content about Israel, Israeli politics, Judaism, Jewish culture, or Israeli-Palestinian affairs"),
    123302: ("Jewish religious discourse", "Explanatory text about Jewish laws, practices, beliefs, and theological concepts in religious instruction contexts"),
    14127: ("Jewish surnames", "Ashkenazi Jewish surnames and names in biographical, legal, and genealogical contexts"),
}

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = "cuda"

# Load model and SAE
model, tokenizer = load_model_and_tokenizer(BASE_MODEL, HF_MODEL_ID)
model.eval()
sae = load_sae_for_layer(SAE_REPO, ANALYSIS_LAYER, TRAINER, device=device)

# Load dataset
dataset = load_dataset(DATASET_NAME, "main", split="train")
if NUM_SAMPLES < len(dataset):
    indices = random.sample(range(len(dataset)), NUM_SAMPLES)
    sampled_data = dataset.select(indices)
else:
    sampled_data = dataset
questions = [item[QUESTION_FIELD] for item in sampled_data]

# Generate date-varied prompts
def create_dated_prompt(question: str, year: int, month_day: str) -> str:
    return f"Today is {year}-{month_day}. {question}"

days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
base_dates = []
for _ in questions:
    month = random.randint(1, 12)
    day = random.randint(1, days_in_month[month - 1])
    base_dates.append(f"{month:02d}-{day:02d}")

prompts_by_year = {
    year: [create_dated_prompt(q, year, d) for q, d in zip(questions, base_dates)]
    for year in YEARS
}

# Extract activations
def prompts_to_dataset_rows(prompts):
    return [{"messages": [{"role": "user", "content": p}]} for p in prompts]

activations_by_year = {}
for year in YEARS:
    activations = extract_activations_batched(
        model=model,
        tokenizer=tokenizer,
        dataset_rows=prompts_to_dataset_rows(prompts_by_year[year]),
        layers=[ANALYSIS_LAYER],
        batch_size=BATCH_SIZE,
        max_ctx_len=MAX_CTX_LEN,
        device=device,
        dtype=torch.float32
    )
    activations_by_year[year] = activations[ANALYSIS_LAYER]

# Compute mean differences and identify top features
activation_diff = activations_by_year[TARGET_YEAR].mean(dim=0) - activations_by_year[REFERENCE_YEAR].mean(dim=0)

feature_cosine_sims = compute_feature_cosine_similarities(
    activation_diff.to(device),
    sae,
    device=device
)

top_k_values, top_k_indices = torch.topk(feature_cosine_sims, TOP_K_FEATURES)
top_features = top_k_indices.cpu().detach().numpy()

# Compute feature projections across years
def project_onto_feature_decoder(activations, feature_idx, sae):
    decoder_weight = sae.decoder.weight
    feature_dir = decoder_weight[:, feature_idx] if decoder_weight.shape[0] == activations.shape[1] else decoder_weight[feature_idx, :]
    feature_dir_norm = (feature_dir / feature_dir.norm()).to(device)
    return einops.einsum(activations.to(device), feature_dir_norm, "b d, d -> b").cpu()

feature_projections = {}
for feature_idx in tqdm(top_features, desc="Computing projections"):
    feature_projections[feature_idx] = {
        year: project_onto_feature_decoder(activations_by_year[year], feature_idx, sae)
        for year in YEARS
    }

# Plot results
mean_projections = {
    fid: {year: feature_projections[fid][year].mean().item() for year in YEARS}
    for fid in top_features
}

def format_feature_label(feature_idx):
    if feature_idx in FEATURE_LABELS:
        return f"F{feature_idx} ({FEATURE_LABELS[feature_idx][0]})"
    return f"F{feature_idx}"

fig, ax = plt.subplots(figsize=(12, 8))
year_labels = [str(year) for year in YEARS]

for feature_idx in top_features:
    means = [mean_projections[feature_idx][year] for year in YEARS]
    ax.plot(year_labels, means, marker='o', linewidth=2, markersize=8,
            label=format_feature_label(feature_idx), alpha=0.7)

target_year_idx = YEARS.index(TARGET_YEAR)
ax.axvspan(target_year_idx - 0.5, target_year_idx + 0.5, alpha=0.1, color='gray')

ax.set_xlabel('Year', fontsize=14)
ax.set_ylabel('Mean Feature Projection', fontsize=14)
ax.set_title(f'Top {TOP_K_FEATURES} Features: Mean Projections Across Years', fontsize=16)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

OUTPUT_DIR = Path(__file__).parent / "results" / "identify_features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / "feature_projections.pdf"
plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
print(f"Plot saved: {output_path}")

plt.show()
