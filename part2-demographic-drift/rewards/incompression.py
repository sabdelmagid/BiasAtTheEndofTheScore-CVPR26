import math

import torch
import torch.nn.functional as F

from rewards.base_reward import BaseRewardLoss


def laplacian_energy(img: torch.Tensor) -> torch.Tensor:
    if img.ndim == 3:
        img = img.unsqueeze(0)
    gray = img.mean(dim=1, keepdim=True)
    lap = torch.tensor([[0, 1, 0],
                        [1,-4, 1],
                        [0, 1, 0]], device=img.device, dtype=img.dtype).view(1, 1, 3, 3)
    filtered = F.conv2d(gray, lap, padding=1)
    return filtered.abs().mean()


def frequency_energy(img: torch.Tensor, high_freq_frac: float = 0.1) -> torch.Tensor:
    """
    Compute average high-frequency energy in a CLIP-normalized image tensor.
    Args:
        img: [B, 3, H, W] or [3, H, W], CLIP-normalized
        high_freq_frac: fraction of spectrum edges to treat as "high frequency"
    Returns:
        Scalar tensor (float32)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)

    gray = img.mean(dim=1, keepdim=True).to(torch.float32)  # <-- Cast to float32 for FFT

    freq = torch.fft.fftshift(torch.fft.fft2(gray, norm="ortho"))
    mag = freq.abs()

    H, W = mag.shape[-2:]
    center_h, center_w = H // 2, W // 2
    radius_h, radius_w = int(H * high_freq_frac / 2), int(W * high_freq_frac / 2)
    mask = torch.ones_like(mag)
    mask[..., center_h - radius_h:center_h + radius_h,
                center_w - radius_w:center_w + radius_w] = 0.0

    high_freq_energy = (mag * mask).mean()
    return high_freq_energy.to(img.dtype)  # cast back if needed


def _dct_matrix(N: int, device, dtype):
    # Orthonormal DCT-II matrix (NxN)
    k = torch.arange(N, device=device, dtype=dtype).view(1, -1)
    n = torch.arange(N, device=device, dtype=dtype).view(-1, 1)
    C = torch.cos(math.pi * (n + 0.5) * k / N)
    C[:, 0] = C[:, 0] / math.sqrt(2.0)
    C = C * math.sqrt(2.0 / N)
    return C  # (N,N)

def _make_hf_mask_8x8(hf_min_sum: int = 6, device="cpu", dtype=torch.bool):
    # High-frequency mask on 8x8 (True where u+v >= hf_min_sum)
    u = torch.arange(8, device=device).view(-1, 1)
    v = torch.arange(8, device=device).view(1, -1)
    return (u + v) >= hf_min_sum  # (8,8) bool

@torch.cuda.amp.autocast(enabled=False)  # keep this in fp32 for stability
def block_dct_highfreq_sparsity(
    img: torch.Tensor,
    hf_min_sum: int = 6,
    stride: int = 8,
    thresh: float = 0.02,
    sharpness: float = 50.0,
) -> torch.Tensor:
    """
    Strong JPEG-like penalty: soft count of non-zero high-frequency 8x8 DCT coeffs.
    Args:
        img: [B,3,H,W] or [3,H,W], CLIP-normalized (works as-is)
        hf_min_sum: select (u,v) with u+v >= hf_min_sum as high-frequency bins
        stride: block stride; 8 = non-overlap, 4 = overlap (stronger)
        thresh: magnitude threshold for "non-zero" (softly)
        sharpness: sigmoid slope; larger -> closer to hard count (L0)
    Returns:
        Scalar tensor: higher => more high-freq nonzeros (less compressible)
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)          # [1,3,H,W]
    B, C, H, W = img.shape
    x = img.mean(1, keepdim=True).to(torch.float32)  # grayscale, fp32

    # Unfold into 8x8 blocks (optionally overlapping via stride)
    unfold = F.unfold(x, kernel_size=8, stride=stride)  # [B, 8*8, L]
    L = unfold.shape[-1]
    blocks = unfold.transpose(1, 2).contiguous().view(B*L, 8, 8)  # [B*L,8,8]

    # 2D DCT via C @ block @ C^T
    C8 = _dct_matrix(8, device=blocks.device, dtype=blocks.dtype)  # (8,8)
    dct = C8 @ blocks @ C8.t()  # [B*L,8,8]

    # Select high-frequency bins
    hf_mask = _make_hf_mask_8x8(hf_min_sum, device=blocks.device)  # (8,8)
    hf = dct[:, hf_mask]  # [B*L, #hf]

    # Soft L0: ~ count(|c| > thresh)
    # sigmoid(k*(|c|-thresh)) ~ 1 if |c| > thresh else 0
    s = torch.sigmoid(sharpness * (hf.abs() - thresh))

    # Average over all coeffs & blocks
    sparsity = s.mean()  # higher => more high-freq nonzeros (bad if you want smooth)
    return sparsity

def compression_proxy_from_clip(img: torch.Tensor) -> torch.Tensor:
    """
    A strong, JPEG-like proxy to *minimize* if you want low-frequency images.
    """
    return block_dct_highfreq_sparsity(
        img,
        hf_min_sum=6,   # tighten to 7 for even stronger HF-only
        stride=8,       # use 4 for overlapping (stronger)
        thresh=0.02,    # increase if you want to ignore tiny coefficients
        sharpness=50.0  # 30–100; larger ~ harder threshold
    )

# -------------------------------
# Compression loss using CLIP-normalized input
# -------------------------------
class IncompressionLoss(BaseRewardLoss):
    """Compression reward/loss using a differentiable JPEG 'size' proxy."""

    def __init__(
        self,
        weighting: float,
        dtype: torch.dtype,
        device: torch.device,
        cache_dir: str,
        memsave: bool = False,
    ):
        super().__init__("Incompression", weighting)
        self.device = device
        

    
    @torch.no_grad()
    def _debug_print(self, val: torch.Tensor, tag: str):
        try:
            print("*" * 10)
            print(f"{tag} =", float(val))
            print("*" * 10)
        except Exception:
            pass

    def get_image_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        image: CLIP-normalized tensor in B x 3 x H x W  (roughly [-2, +2])
        Returns: scalar proxy ~ JPEG file size (higher = less compressible)
        """
        e = compression_proxy_from_clip(image)
        #self._debug_print(e, 'lap energy')
        #e = 1 - e
        #self._debug_print(e, '1-lap energy')
        
        return e

    def get_text_features(self, prompt: str) -> torch.Tensor:
        return None

    def compute_loss(self, image_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        
        return 1-image_features

    def __call__(self, image: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        image_features = self.get_image_features(image)
        return self.compute_loss(image_features, None)
