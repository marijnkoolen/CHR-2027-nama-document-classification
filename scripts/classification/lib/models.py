"""Model builders shared by both pipelines under scripts/classification/.

Both single-image scenarios (joint/train.py's page mode) and the
sequence-context scenario (joint/train.py's sequence mode) use the same
recipe for turning an image into a feature vector - a pretrained ViT-style
backbone loaded via `transformers.AutoModel`, [CLS]-token pooled - just
with a different backbone size and freeze/fine-tune policy. This works for
DINOv2 checkpoints (Dinov2Model) and DiT checkpoints (BeitModel, since DiT
reuses the BEiT architecture) without needing separate code paths.
`PageEmbedder` holds that shared logic; `BackboneClassifier` adds a plain
classification head on top of it for single-image use, and
sequence_model.py attaches a page-embedder to a sequence-context head
instead.

`ConvPageEmbedder` is the same idea for torchvision CNN backbones (VGG16,
EfficientNet-B0) that aren't loadable via `transformers.AutoModel` at all -
see its own docstring. `build_image_embedder()` dispatches between it and
PageEmbedder by backbone name, so every caller that used to construct
`PageEmbedder(...)` directly goes through it instead and transparently
supports both backbone families with no caller-side branching.

`TextEmbedder` is the same idea for a page's transcribed text, and
`MultimodalPageEmbedder` late-fuses image+text by concatenating their
embeddings.

build_vgg16_classifier / build_efficientnet_classifier: a second,
DELIBERATELY DIFFERENT end-to-end image fine-tuning recipe for those same
two backbones - scripts/classification/sequential/'s own bespoke freeze
policy (VGG16: freeze all conv, fine-tune only the original 4096-D FC
classifier head; EfficientNet-B0: freeze only the stem + first MBConv
stage), kept from that pipeline's origin. This is NOT interchangeable with
ConvPageEmbedder/BackboneClassifier for the same two backbone names -
ConvPageEmbedder drops VGG16's FC classifier entirely (global-average-pools
the conv feature map instead) and uses a generic, parametrized
unfreeze_last_n_blocks conv-stage policy; these two builders keep VGG16's
original FC path and each backbone's own fixed policy. Both exist because
scripts/classification/sequential/'s results were produced with these
specific architectures - build_image_classifier() below routes "vgg16"/
"efficientnet_b0" to these bespoke builders and everything else (any other
HuggingFace checkpoint) to BackboneClassifier, so both conventions coexist
without either silently overriding the other.

LSTMClassifier: per-page classification over a whole dossier's cached
feature sequence (sequential/train_sequence.py, any backbone - see
lib/embeddings.py).

TextCNN / BERTClassifier: text-only fine-tuning (sequential/train_finetune.py).

EarlyFusionMLP: projects a frozen image-backbone feature vector and a
frozen text-backbone feature vector into a shared hidden size, concatenates,
then an MLP head (sequential/train_fusion.py).
"""

from __future__ import annotations

import inspect

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


def trainable_parameter_summary(module: nn.Module) -> str:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return f"{trainable:,} / {total:,} parameters trainable ({100 * trainable / total:.1f}%)"


def trainable_param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _find_transformer_blocks(model: nn.Module) -> nn.ModuleList:
    """Locate the list of transformer blocks regardless of backbone family
    (Dinov2Model: model.encoder.layer; BeitModel/DiT: model.layers;
    XLMRobertaModel and friends: model.encoder.layer)."""
    for path in ("encoder.layer", "layers", "encoder.layers", "blocks"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        if isinstance(obj, nn.ModuleList):
            return obj
    raise ValueError(f"Could not find transformer blocks on {type(model).__name__}")


def _attn_implementation_kwargs(device: torch.device | None) -> dict:
    """MPS's scaled_dot_product_attention kernel doesn't support dropout (as
    of this PyTorch version) - hit during training, since attention dropout
    is only active in train() mode. HF's default 'sdpa' attention
    implementation uses that kernel and so errors there; fall back to the
    slower but backend-agnostic 'eager' implementation on MPS only, leaving
    the faster/more memory-efficient SDPA path on CUDA/CPU."""
    if device is not None and device.type == "mps":
        return {"attn_implementation": "eager"}
    return {}


def _freeze_all_but_last_n(
    backbone: nn.Module, unfreeze_last_n: int, gradient_checkpointing: bool = False
) -> None:
    for p in backbone.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        blocks = _find_transformer_blocks(backbone)
        for block in blocks[-unfreeze_last_n:]:
            for p in block.parameters():
                p.requires_grad = True
        # Only useful (and only takes effect) when something is actually
        # unfrozen: a fully frozen backbone builds no backward graph through
        # itself at all (no output here requires grad, so autograd already
        # skips retaining its activations) - checkpointing has nothing to
        # save memory on there, it would just recompute forward for nothing.
        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable()


class PageEmbedder(nn.Module):
    """Pretrained ViT-style backbone -> single embedding per image, with a
    configurable number of trailing transformer blocks left trainable
    (0 = frozen, linear-probe style).

    project_to: optionally project the backbone's native embedding down to a
    smaller size. Matters most for sequence mode: SequenceContextModel's
    heads scale with embed_dim (doc_in = embed_dim*2), so a 1024-dim backbone
    (DiT-large) gives it ~7x the parameters a 384-dim one (DINOv2-small)
    would - badly overparameterized against a PDF-level training set that's
    typically just a few dozen documents, which in practice collapses
    training onto the trivial "predict the label's marginal frequency,
    ignore the page" solution regardless of learning rate or model depth.
    Projecting down to something in DINOv2-small's range (e.g. 384) before
    the sequence model fixes this without giving up the larger backbone's
    features entirely."""

    def __init__(self, backbone_name: str, unfreeze_last_n_blocks: int = 0, device: torch.device | None = None,
                 gradient_checkpointing: bool = False, project_to: int | None = None):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name, **_attn_implementation_kwargs(device))
        backbone_dim = self.backbone.config.hidden_size
        # Some backbones (Beit/DiT) need an explicit flag to interpolate their
        # position embeddings for input sizes other than the pretraining
        # resolution; others (Dinov2) handle arbitrary sizes automatically.
        self._supports_pos_interp = "interpolate_pos_encoding" in inspect.signature(
            self.backbone.forward
        ).parameters
        # Beit-family checkpoints (DiT included) trained with
        # use_mean_pooling=True read out via a separate pooler (LayerNorm
        # over mean-pooled patch tokens, `pooler_output`) rather than the
        # [CLS] token - for those, last_hidden_state[:, 0] is essentially an
        # untrained, unstable readout (seen empirically: std ~70 vs ~1 for
        # the pooler, occasionally spiking into the thousands and blowing up
        # downstream LayerNorms/losses). Dinov2 and plain ViT checkpoints
        # don't set this flag and use [CLS] as intended.
        self._use_pooler = bool(getattr(self.backbone.config, "use_mean_pooling", False))
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n_blocks, gradient_checkpointing)

        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = backbone_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        kwargs = {"interpolate_pos_encoding": True} if self._supports_pos_interp else {}
        outputs = self.backbone(pixel_values=pixel_values, **kwargs)
        cls = outputs.pooler_output if self._use_pooler else outputs.last_hidden_state[:, 0]
        return self.projection(cls)


CNN_BACKBONES = {
    "vgg16": {"out_dim": 512},
    "efficientnet_b0": {"out_dim": 1280},
}


def _vgg16_stages(features: nn.Module) -> list[nn.Module]:
    """VGG16's `.features` has no native grouping - just a flat stack of
    Conv2d/ReLU/MaxPool2d layers - so stages are delimited at each
    MaxPool2d boundary (5 stages, matching VGG's own architecture diagram).
    Each stage is wrapped in a throwaway nn.Sequential purely so
    `.parameters()` can iterate it in one call below - the wrapper
    references the exact same layer instances already registered under
    `features` (not copies), so toggling requires_grad through it still
    affects the real forward-pass module."""
    stages, current = [], []
    for layer in features:
        current.append(layer)
        if isinstance(layer, nn.MaxPool2d):
            stages.append(nn.Sequential(*current))
            current = []
    if current:
        stages.append(nn.Sequential(*current))
    return stages


def _freeze_all_but_last_n_stages(features: nn.Module, stages: list[nn.Module], unfreeze_last_n: int) -> None:
    for p in features.parameters():
        p.requires_grad = False
    if unfreeze_last_n > 0:
        for stage in stages[-unfreeze_last_n:]:
            for p in stage.parameters():
                p.requires_grad = True


class ConvPageEmbedder(nn.Module):
    """Pretrained torchvision CNN backbone -> single embedding per image -
    the CNN-family counterpart to PageEmbedder, which only handles
    HuggingFace transformer checkpoints (loaded via AutoModel and pooled
    via [CLS]/pooler_output - neither concept applies to a CNN). Exposes
    the identical contract (.embed_dim, forward(pixel_values) -> (B,
    embed_dim)) so it drops into BackboneClassifier/MultimodalPageEmbedder/
    SequenceContextModel wherever a PageEmbedder is otherwise used - see
    build_image_embedder() below for the dispatch between the two, keyed
    on backbone name.

    Pooling: global-average-pool over the final conv feature map (matching
    EfficientNet's own classifier head, and the standard convnet embedding
    convention) - not VGG16's original 4096-D fc7 layer (dropped along
    with the rest of the ImageNet classifier), since that FC path's
    dropout/weights were fit for 1000-way ImageNet classification
    specifically, not as a general-purpose embedding. (Contrast with this
    same file's build_vgg16_classifier, a separate builder that keeps that
    FC path - see this module's docstring for why both exist.)

    unfreeze_last_n_blocks: the CNN analogue of PageEmbedder's "last N
    transformer blocks" - here, "last N conv stages" (VGG16: 5 stages, see
    _vgg16_stages; EfficientNet-B0: its 9 native `.features` children -
    stem, 7 MBConv stages, head conv - used directly as stages, since
    `.parameters()` already recurses through whatever's nested inside
    each one).

    Preprocessing: uses the exact same ImageNet mean/std normalization as
    every HuggingFace ViT-style backbone here (see common.py's
    build_transforms) - already compatible, no separate preprocessing path
    needed.

    gradient_checkpointing has no effect here (accepted for signature
    compatibility with PageEmbedder/build_image_embedder only): torchvision
    CNNs don't expose the same checkpointing API as HF transformers, and
    these backbones are small enough not to need it."""

    SUPPORTED = tuple(CNN_BACKBONES)

    def __init__(self, backbone_name: str, unfreeze_last_n_blocks: int = 0,
                 device: torch.device | None = None, gradient_checkpointing: bool = False,
                 project_to: int | None = None):
        super().__init__()
        if backbone_name not in CNN_BACKBONES:
            raise ValueError(f"unknown CNN backbone {backbone_name!r} - expected one of {self.SUPPORTED}")

        from torchvision import models as tv_models

        if backbone_name == "vgg16":
            backbone = tv_models.vgg16(weights=tv_models.VGG16_Weights.IMAGENET1K_V1)
            self.features = backbone.features
            stages = _vgg16_stages(self.features)
        else:
            backbone = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.features = backbone.features
            stages = list(self.features.children())

        self.pool = nn.AdaptiveAvgPool2d(1)
        _freeze_all_but_last_n_stages(self.features, stages, unfreeze_last_n_blocks)

        backbone_dim = CNN_BACKBONES[backbone_name]["out_dim"]
        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = backbone_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pooled = torch.flatten(self.pool(self.features(pixel_values)), 1)
        return self.projection(pooled)


def build_image_embedder(
    backbone_name: str, unfreeze_last_n_blocks: int = 0, device: torch.device | None = None,
    gradient_checkpointing: bool = False, project_to: int | None = None,
) -> nn.Module:
    """PageEmbedder (HuggingFace transformer checkpoints) or ConvPageEmbedder
    (torchvision CNN checkpoints), chosen by backbone_name - the single
    place that decision is made, so BackboneClassifier/MultimodalPageEmbedder/
    joint/train.py's sequence-mode vision branch don't each need their own
    if/else for it."""
    if backbone_name in CNN_BACKBONES:
        return ConvPageEmbedder(
            backbone_name, unfreeze_last_n_blocks, device=device,
            gradient_checkpointing=gradient_checkpointing, project_to=project_to,
        )
    return PageEmbedder(
        backbone_name, unfreeze_last_n_blocks, device=device,
        gradient_checkpointing=gradient_checkpointing, project_to=project_to,
    )


class BackboneClassifier(nn.Module):
    """A PageEmbedder/ConvPageEmbedder + linear head, for plain single-image classification."""

    def __init__(self, backbone_name: str, num_classes: int, unfreeze_last_n_blocks: int = 0,
                 device: torch.device | None = None, gradient_checkpointing: bool = False,
                 project_to: int | None = None):
        super().__init__()
        self.embedder = build_image_embedder(backbone_name, unfreeze_last_n_blocks, device=device,
                                              gradient_checkpointing=gradient_checkpointing, project_to=project_to)
        self.head = nn.Sequential(nn.LayerNorm(self.embedder.embed_dim), nn.Linear(self.embedder.embed_dim, num_classes))

    @property
    def backbone(self):
        return self.embedder.backbone

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(pixel_values))


def build_vgg16_classifier(num_classes: int, device: torch.device) -> nn.Module:
    """scripts/classification/sequential/'s bespoke VGG16 recipe: freezes
    all convolutional feature layers, fine-tunes only the FC classifier
    head (with its final layer resized to num_classes) - keeps VGG16's
    original 4096-D fc7 path, unlike ConvPageEmbedder/BackboneClassifier's
    "vgg16" (see this module's docstring for why both exist)."""
    from torchvision import models as tv_models

    weights = tv_models.VGG16_Weights.IMAGENET1K_V1
    model = tv_models.vgg16(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model.to(device)


def build_efficientnet_classifier(num_classes: int, device: torch.device) -> nn.Module:
    """scripts/classification/sequential/'s bespoke EfficientNet-B0 recipe:
    freezes only the first two feature blocks (stem + first MBConv stage),
    fine-tunes the rest plus a classifier head resized to num_classes - see
    this module's docstring for why this differs from ConvPageEmbedder/
    BackboneClassifier's "efficientnet_b0"."""
    from torchvision import models as tv_models

    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = tv_models.efficientnet_b0(weights=weights)
    for name, p in model.named_parameters():
        if "features.0" in name or "features.1" in name:
            p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)


def build_image_classifier(
    backbone_name: str, num_classes: int, device: torch.device, unfreeze_last_n_blocks: int = 2,
) -> nn.Module:
    """scripts/classification/sequential/'s single dispatch point for
    end-to-end image fine-tuning: "vgg16"/"efficientnet_b0" go to their
    bespoke builders above (fixed freeze policy each); any other string is
    treated as a HuggingFace checkpoint and goes to BackboneClassifier
    (unfreeze_last_n_blocks applies - see sequential/train_finetune.py's
    --unfreeze-blocks)."""
    if backbone_name == "vgg16":
        return build_vgg16_classifier(num_classes, device)
    if backbone_name in ("efficientnet", "efficientnet_b0"):
        return build_efficientnet_classifier(num_classes, device)
    return BackboneClassifier(
        backbone_name, num_classes, unfreeze_last_n_blocks=unfreeze_last_n_blocks, device=device,
    ).to(device)


class TextEmbedder(nn.Module):
    """Pretrained multilingual transformer -> single embedding per page's
    transcribed text, mean-pooled over non-padding tokens (a better
    off-the-shelf sentence representation than the [CLS]/<s> token when the
    backbone is frozen or only lightly fine-tuned).

    project_to: same reasoning as PageEmbedder's - matters most for
    text-only sequence mode, where SequenceContextModel's heads scale with
    embed_dim (doc_in = embed_dim*2), so a large text backbone (e.g.
    XLM-R-large, 1024-dim) can overparameterize them the same way a large
    vision backbone did."""

    def __init__(self, backbone_name: str = "xlm-roberta-base", unfreeze_last_n_layers: int = 0,
                 max_length: int = 256, device: torch.device | None = None, gradient_checkpointing: bool = False,
                 project_to: int | None = None):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.backbone = AutoModel.from_pretrained(backbone_name, **_attn_implementation_kwargs(device))
        backbone_dim = self.backbone.config.hidden_size
        self.max_length = max_length
        _freeze_all_but_last_n(self.backbone, unfreeze_last_n_layers, gradient_checkpointing)

        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(backbone_dim), nn.Linear(backbone_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = backbone_dim

    def tokenize(self, texts: list[str]):
        return self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).to(outputs.last_hidden_state.dtype)
        summed = (outputs.last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        return self.projection(pooled)


class TextBackboneClassifier(nn.Module):
    """A TextEmbedder + linear head, for plain single-page text-only
    classification - the text-only analogue of BackboneClassifier."""

    def __init__(self, backbone_name: str, num_classes: int, unfreeze_last_n_layers: int = 0,
                 max_length: int = 256, device: torch.device | None = None, gradient_checkpointing: bool = False,
                 project_to: int | None = None):
        super().__init__()
        self.embedder = TextEmbedder(backbone_name, unfreeze_last_n_layers, max_length, device=device,
                                      gradient_checkpointing=gradient_checkpointing, project_to=project_to)
        self.head = nn.Sequential(nn.LayerNorm(self.embedder.embed_dim), nn.Linear(self.embedder.embed_dim, num_classes))

    @property
    def backbone(self):
        return self.embedder.backbone

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(input_ids, attention_mask))


class MultimodalPageEmbedder(nn.Module):
    """Late fusion of PageEmbedder (image) and TextEmbedder (transcribed
    text): each modality is embedded independently and the two vectors are
    concatenated (optionally projected back down to a chosen size), so this
    exposes the same fixed-size `embed_dim` output as a plain PageEmbedder
    and can be dropped in wherever one is expected."""

    def __init__(
        self,
        image_backbone: str,
        text_backbone: str = "xlm-roberta-base",
        unfreeze_image_blocks: int = 0,
        unfreeze_text_layers: int = 0,
        max_text_length: int = 256,
        project_to: int | None = None,
        device: torch.device | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.image_embedder = build_image_embedder(image_backbone, unfreeze_image_blocks, device=device,
                                                    gradient_checkpointing=gradient_checkpointing)
        self.text_embedder = TextEmbedder(text_backbone, unfreeze_text_layers, max_text_length, device=device,
                                           gradient_checkpointing=gradient_checkpointing)

        combined_dim = self.image_embedder.embed_dim + self.text_embedder.embed_dim
        if project_to:
            self.projection = nn.Sequential(nn.LayerNorm(combined_dim), nn.Linear(combined_dim, project_to), nn.GELU())
            self.embed_dim = project_to
        else:
            self.projection = nn.Identity()
            self.embed_dim = combined_dim

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        image_embed = self.image_embedder(pixel_values)
        text_embed = self.text_embedder(input_ids, attention_mask)
        return self.projection(torch.cat([image_embed, text_embed], dim=-1))


class MultimodalBackboneClassifier(nn.Module):
    """A MultimodalPageEmbedder + linear head."""

    def __init__(
        self,
        image_backbone: str,
        num_classes: int,
        text_backbone: str = "xlm-roberta-base",
        unfreeze_image_blocks: int = 0,
        unfreeze_text_layers: int = 0,
        max_text_length: int = 256,
        project_to: int | None = None,
        device: torch.device | None = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.embedder = MultimodalPageEmbedder(
            image_backbone, text_backbone, unfreeze_image_blocks, unfreeze_text_layers, max_text_length, project_to,
            device=device, gradient_checkpointing=gradient_checkpointing,
        )
        self.head = nn.Sequential(nn.LayerNorm(self.embedder.embed_dim), nn.Linear(self.embedder.embed_dim, num_classes))

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedder(pixel_values, input_ids, attention_mask))


class LSTMClassifier(nn.Module):
    def __init__(
        self, input_dim: int = 4096, hidden_dim: int = 256, num_layers: int = 2,
        num_classes: int = 2, dropout: float = 0.3,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(
            input_size=512, hidden_size=hidden_dim, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0, bidirectional=True,
        )
        self.head = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        x = self.proj(x)  # (B, T, 512)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(x)
        if lengths is not None:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return self.head(out)  # (B, T, num_classes)


class TextCNN(nn.Module):
    """Frozen BERT token embeddings -> multi-scale 1D-CNN -> max-over-time
    pool -> linear head."""

    def __init__(
        self, bert_model: nn.Module, embed_dim: int = 768, num_filters: int = 128,
        filter_sizes: tuple = (2, 3, 4), num_classes: int = 2, dropout: float = 0.4,
    ):
        super().__init__()
        self.bert = bert_model  # frozen by the caller
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            emb = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # (B, T, E)
        x = emb.transpose(1, 2)  # (B, E, T)
        pooled = [
            F.max_pool1d(F.relu(conv(x)), x.size(2) - conv.kernel_size[0] + 1).squeeze(2) for conv in self.convs
        ]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class BERTClassifier(nn.Module):
    """End-to-end BERT fine-tuning: [CLS] hidden state -> dropout -> linear head."""

    def __init__(self, model_name: str = "bert-base-uncased", num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.bert.config.hidden_size, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])


class EarlyFusionMLP(nn.Module):
    def __init__(
        self, img_dim: int, text_dim: int = 768, hidden: int = 512,
        num_classes: int = 2, dropout: float = 0.3,
    ):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden)
        self.text_proj = nn.Linear(text_dim, hidden)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, img_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.img_proj(img_feat), self.text_proj(text_feat)], dim=1))
