import torch
from transformers import CLIPModel
from rewards.base_reward import BaseRewardLoss


class ContrastiveCLIPLoss(BaseRewardLoss):
    """CLIP reward that can run in contrastive or positive-only mode."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        tokenizer,
        pos_prompt: str,
        neg_prompt: str = None,
        alpha: float = 1.0,
        contrastive: bool = True,
        memsave: bool = False,
    ):
        """
        Args:
            weighting: scalar multiplier for total loss
            dtype: torch dtype (e.g., torch.float16)
            device: torch device
            cache_dir: where to store CLIP weights
            tokenizer: same tokenizer used elsewhere
            pos_prompt: positive prompt (e.g., "A photo of a Black doctor")
            neg_prompt: negative prompt (optional, e.g., "A photo of a White doctor")
            alpha: how strongly to push away from negative prompt (only used if contrastive=True)
            contrastive: if False, use only positive similarity objective
            memsave: optional low-memory conversion
        """
        self.tokenizer = tokenizer
        self.clip_model = CLIPModel.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
        )

        if memsave:
            import memsave_torch.nn
            self.clip_model = memsave_torch.nn.convert_to_memory_saving(self.clip_model)

        self.clip_model = self.clip_model.to(device, dtype=dtype)
        self.clip_model.eval()
        self.freeze_parameters(self.clip_model.parameters())
        self.clip_model.gradient_checkpointing_enable()

        # Cache text features for efficiency
        self.pos_text_features = self.get_text_features(pos_prompt)
        self.neg_text_features = None
        if contrastive and neg_prompt is not None:
            self.neg_text_features = self.get_text_features(neg_prompt)

        self.alpha = alpha
        self.contrastive = contrastive
        super().__init__("ContrastiveCLIP" if contrastive else "PositiveCLIP", weighting)

    def get_text_features(self, prompt: str) -> torch.Tensor:
        tokenized = self.tokenizer(
            prompt, return_tensors="pt", padding=True, max_length=77, truncation=True
        ).to(next(self.clip_model.parameters()).device)
        with torch.no_grad():
            return self.clip_model.get_text_features(**tokenized).detach()

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        return self.clip_model.get_image_features(image)

    def compute_loss(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Contrastive (default): L = 100 - (sim_pos - α * sim_neg) * logit_scale
        Positive-only:         L = 100 - sim_pos * logit_scale
        """
        img = image_features / image_features.norm(dim=-1, keepdim=True)
        pos = self.pos_text_features / self.pos_text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        sim_pos = (img @ pos.T).mean()

        if self.contrastive and self.neg_text_features is not None:
            neg = self.neg_text_features / self.neg_text_features.norm(dim=-1, keepdim=True)
            sim_neg = (img @ neg.T).mean()
            clip_loss = 100 - (sim_pos - self.alpha * sim_neg) * logit_scale
        else:
            clip_loss = 100 - sim_pos * logit_scale

        return clip_loss

    def __call__(self, image: torch.Tensor, prompt: str = None) -> torch.Tensor:
        image_features = self.get_image_features(image)
        return self.compute_loss(image_features)
