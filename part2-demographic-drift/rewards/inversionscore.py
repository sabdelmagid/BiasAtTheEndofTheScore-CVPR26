"""Multi-reference CLIP inversion objective."""

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel

from rewards.base_reward import BaseRewardLoss

class InversionLoss(BaseRewardLoss):
    """
    CLIP + Text inversion loss (multi-reference version).

    Combines:
      (1) CLIP image–image similarity  → semantic inversion alignment (multi-ref)
      (2) CLIP image–text similarity   → textual semantic guidance (multi-text)
    """

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        tokenizer=None,  # not used externally
        memsave: bool = False,
        text_weight: float = 0.25,     # weight for image–text alignment
        #text_prompts: list = None,     # list of textual guidance prompts
        args =  None
    ):
        super().__init__("Inversion", weighting)

        # ---- Load CLIP backbone ----
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

        # ---- Tokenizer for text encoding ----
        self.tokenizer = AutoTokenizer.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            cache_dir=cache_dir,
        )

        self.device = device
        self.dtype = dtype
        self.text_weight = text_weight

        # ---- Hardcoded reference image paths ----
        reference_image_paths = args.inversion_target_images

    
        # ---- CLIP preprocessing ----
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

        # ---- Load and encode reference images ----
        self.reference_image_tensors = []
        self.reference_image_features = []

        for path in reference_image_paths:
            try:
                ref_img = Image.open(path).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Reference image not found at {path}. Please check the path.")

            ref_tensor = self.preprocess(ref_img).unsqueeze(0).to(device=device, dtype=dtype)
            self.reference_image_tensors.append(ref_tensor)

            with torch.no_grad():
                feats = self.clip_model.get_image_features(ref_tensor)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                self.reference_image_features.append(feats)

        self.reference_image_features = torch.cat(self.reference_image_features, dim=0)  # [N, D]

        # ---- Precompute text embeddings (multi-text) ----
        text_prompts = args.inversion_target_texts
        self.negative_text_prompts = args.inversion_target_negative_texts
        print("Text prompts used for anchoring -> ", text_prompts)
            

        self.text_features = []
        with torch.no_grad():
            for prompt in text_prompts:
                text_tokens = self.tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
                ).to(device)
                text_feats = self.clip_model.get_text_features(**text_tokens)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                self.text_features.append(text_feats)

        self.text_features = torch.cat(self.text_features, dim=0)  # [M, D]

        if self.negative_text_prompts:
            self.negative_text_features = []
            with torch.no_grad():
                for prompt in self.negative_text_prompts:
                    text_tokens = self.tokenizer(
                        prompt, return_tensors="pt", padding=True, truncation=True, max_length=77
                    ).to(device)
                    text_feats = self.clip_model.get_text_features(**text_tokens)
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                    self.negative_text_features.append(text_feats)
                    
            self.negative_text_features = torch.cat(self.negative_text_features, dim=0)  # [M, D]

    # ------------------------------------------------------------------
    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract normalized CLIP image embeddings."""
        img_features = self.clip_model.get_image_features(image)
        return img_features / img_features.norm(dim=-1, keepdim=True)

    def get_text_features(self, prompt: str) -> torch.Tensor:
        return None

    def compute_loss(self, image_features: torch.Tensor) -> torch.Tensor:
        """Compute CLIP losses per reference image and per text prompt, averaged."""
        # (1) CLIP image–image cosine distance per reference
        sims = image_features @ self.reference_image_features.T  # [B, N]
        clip_loss = (1.0 - sims.mean(dim=0)).mean()              # average over references

        # (2) CLIP image–text cosine distance per prompt
        txt_sims = image_features @ self.text_features.T         # [B, M]
        text_loss = (1.0 - txt_sims.mean(dim=0)).mean()          # average over text prompts

        if self.negative_text_prompts:
            txt_sims_neg = image_features @ self.negative_text_features.T         # [B, M]
            text_loss_neg = (1.0 + txt_sims_neg.mean(dim=0)).mean()          # average over text prompts

            text_loss+=text_loss_neg

        return clip_loss, text_loss

    def __call__(self, image: torch.Tensor, prompt: str = None) -> torch.Tensor:
        """
        Compute total loss = CLIP (multi-ref) + multi-text.
        """
        image_features = self.get_image_features(image)
        clip_loss, text_loss = self.compute_loss(image_features)

        total_loss = clip_loss + self.text_weight * text_loss
        return total_loss * self.weighting
