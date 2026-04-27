"""Project paths configuration."""

from pathlib import Path


# Find project root (where pyproject.toml is located)
PROJECT_ROOT = Path(__file__).parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
INTERIM_DATA_DIR = DATA_DIR / 'interim'
SUBMISSION_DATA_DIR = DATA_DIR / 'submission'

# Models directories
MODELS_DIR = PROJECT_ROOT / 'models'
CUSTOM_MODELS_DIR = MODELS_DIR / 'custom'

# TNM stages
STAGES = ['t', 'n', 'm']

STAGES_NUMBERS = {'t': ['1', '2', '3', '4'], 'n': ['0', '1', '2', '3'], 'm': ['0', '1']}

STAGE_LABELS: dict[str, list[str]] = {
    stage: [f'{stage.upper()}{n}' for n in nums]
    for stage, nums in STAGES_NUMBERS.items()
}
