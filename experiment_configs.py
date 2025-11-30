"""
Experiment configurations for ablation study
Run with: python train.py --config_name <name>
"""

CONFIGS = {
    'baseline': {
        'depths': 5,
        'filters_root': 8,
        'use_augmentation': False,
        'temporal_loss_weight': 0.0,
        'description': 'Original PhaseNet architecture'
    },
    
    'augmentation': {
        'depths': 5,
        'filters_root': 8,
        'use_augmentation': True,
        'temporal_loss_weight': 0.0,
        'description': 'Baseline + Data Augmentation'
    },
    
    'temporal_loss': {
        'depths': 5,
        'filters_root': 8,
        'use_augmentation': False,
        'temporal_loss_weight': 0.1,
        'description': 'Baseline + Temporal Consistency Loss'
    },
    
    'deeper': {
        'depths': 6,
        'filters_root': 8,
        'use_augmentation': False,
        'temporal_loss_weight': 0.0,
        'description': 'Deeper network (6 levels)'
    },
    
    'wider': {
        'depths': 5,
        'filters_root': 16,
        'use_augmentation': False,
        'temporal_loss_weight': 0.0,
        'description': 'Wider network (16 filters)'
    },
    
    'full': {
        'depths': 6,
        'filters_root': 16,
        'use_augmentation': True,
        'temporal_loss_weight': 0.1,
        'description': 'All improvements combined'
    }
}

def get_config(name):
    """Get experiment configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Config {name} not found. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]