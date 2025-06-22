import torch
from transformers import AutoModel, AutoTokenizer, ConvNextModel, ConvNextImageProcessor
import os
import torch.nn as nn
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
backbone_vision = "facebook/convnext-large-384"
backbone_text = "vinai/phobert-base"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -------- Vision Encoder (ConvNext Large) --------
class VisionModel(nn.Module):
    def __init__(self, embed_dim = 768):
        super().__init__()
        self.convnext_large = ConvNextModel.from_pretrained(backbone_vision)
        # self.convnext = nn.Sequential(*list(convnext_base.children())[: -2])
        self.projection_head = nn.Linear(1536, embed_dim)
        nn.init.xavier_normal_(self.projection_head.weight)
        nn.init.zeros_(self.projection_head.bias)
        self.drop = nn.Dropout(p = 0.2)

    def forward(self, x):
        x = x.squeeze(1)
        x = self.convnext_large(pixel_values = x)                    
        x = x.pooler_output
        x = self.projection_head(x)
        outputs = self.drop(x)
        return F.normalize(outputs, dim = -1)                      

# -------- Text Encoder (PhoBert base) --------
class TextModel(nn.Module):
    def __init__(self, embed_dim = 768):
        super().__init__()
        self.phobert = AutoModel.from_pretrained(backbone_text)
        self.projection_head = nn.Linear(self.phobert.config.hidden_size, embed_dim)
        nn.init.xavier_normal_(self.projection_head.weight)
        nn.init.zeros_(self.projection_head.bias)
        self.layer_norm = nn.LayerNorm(self.phobert.config.hidden_size)
    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.squeeze(1).to(device)
        attention_mask = attention_mask.squeeze(1).to(device)
        outputs = self.phobert(input_ids = input_ids, attention_mask = attention_mask)
        # CLS token
        cls_embed = outputs.last_hidden_state[:, 0, :]
        projection = self.projection_head(cls_embed)
        outputs = self.layer_norm(projection)
        return F.normalize(outputs, dim = -1)

# -------- VNFOOD-CLIP --------
class Model(nn.Module):
    def __init__(self, embed_dim = 768):
        super().__init__()
        self.vision_encoder = VisionModel(embed_dim = embed_dim)
        self.text_encoder = TextModel(embed_dim=embed_dim)

    def forward(self, images, input_ids, attention_mask):
        image_features = self.vision_encoder(images)
        text_features = self.text_encoder(input_ids, attention_mask)
        return image_features, text_features
    
    def get_image_features(self, images):
        return self.vision_encoder(images)
    
    def get_text_features(self, input_ids, attention_mask):
        return self.text_encoder(input_ids, attention_mask)


def load_model(weight = None):
    processor = ConvNextImageProcessor.from_pretrained(backbone_vision)
    tokenizer = AutoTokenizer.from_pretrained(backbone_text)
    model = Model()
    if weight != None:
        model.load_state_dict(torch.load(weight, map_location = device, weights_only = True))
        print("<All keys matched successfully>")
    return model, processor, tokenizer
