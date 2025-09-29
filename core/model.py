import os, gdown, torch
from torch import nn
from network.models import model_selection
from .config import MODEL_PATH

_model = None
_softmax = None

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        # 기존 gdown ID는 네 코드에 맞춰 바꿔도 됨
        gdown.download(id='1j8AesqDjbSG0RfqaYaHdfGcVVkpdIPKJ', output=MODEL_PATH, quiet=False)

def get_model_and_softmax():
    global _model, _softmax
    if _model is None:
        ensure_model()
        _model = model_selection(modelname='xception', num_out_classes=2)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        _model.eval()
        _softmax = nn.Softmax(dim=1)
    return _model, _softmax
