from models.mlplob import MLPLOB
from models.tlob import TLOB
from models.binctabl import BiN_CTABL
from models.deeplob import DeepLOB
from transformers import AutoModelForSeq2SeqLM

def pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads=8, is_sin_emb=False, dataset_type=None, num_classes=3):
    if model_type == "MLPLOB":
        return MLPLOB(hidden_dim, num_layers, seq_size, num_features, dataset_type, num_classes=num_classes)
    elif model_type == "TLOB":
        return TLOB(hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type, num_classes=num_classes)
    elif model_type == "BINCTABL":
        return BiN_CTABL(60, num_features, seq_size, seq_size, 120, 5, num_classes, 1)
    elif model_type == "DEEPLOB":
        return DeepLOB(num_classes=num_classes)
    else:
        raise ValueError("Model not found")