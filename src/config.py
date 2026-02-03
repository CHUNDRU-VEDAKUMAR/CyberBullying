"""
Project configuration defaults. Consumers can import these or override via CLI.
"""

DEFAULTS = {
    'model_name': 'unitary/toxic-bert',
    'min_score': 0.5,
    'use_lime': False,
    'device': None,  # autodetected if None
    'batch_size': 8,
}
