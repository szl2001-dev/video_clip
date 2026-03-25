"""
VideoCLIPFeatureExtractor
Adapted from /home/work/mllm_datas/yilin/code/video_pipeline/models/video_clip_model.py

Key changes vs. yilin version:
  - model_base_dir is passed explicitly, added to sys.path so imports work
  - preprocess_frames() does NOT flip RGB channels (PyAV returns RGB, model expects RGB)
"""

import sys
from typing import List
import numpy as np
import torch
import torch.nn.functional as F


class VideoCLIPFeatureExtractor:
    def __init__(self, model_path: str, model_base_dir: str, device,
                 torch_dtype: torch.dtype = torch.float32):
        """
        model_path     : path to VideoCLIP-XL-v2.bin
        model_base_dir : directory containing modeling.py + utils/
        device         : torch device string, e.g. 'cuda:0'
        """
        self.device = device

        # Add model directory to sys.path so that its internal imports work
        if model_base_dir not in sys.path:
            sys.path.insert(0, model_base_dir)

        from modeling import VideoCLIP_XL
        from utils.text_encoder import text_encoder as _text_encoder
        self._text_encoder = _text_encoder

        model = VideoCLIP_XL()
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device).eval()
        self.model = model

        # ImageNet normalization stats (RGB order)
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 3, 1, 1)
        self._std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 3, 1, 1)

    # ---------------------------------------------------------------------- #
    # Preprocessing
    # ---------------------------------------------------------------------- #

    def preprocess_frames(self, frames: List[np.ndarray], fnum: int = 8) -> torch.Tensor:
        """
        frames : list of (H, W, 3) np.ndarray in **RGB** format (e.g. from PyAV).
                 Do NOT pass BGR frames (cv2 output) — no channel flip is applied here.
        Returns: (1, T, C, H, W) float tensor on self.device, ready for get_vid_features()
        """
        n = len(frames)
        if n == 0:
            raise ValueError("frames list is empty")

        # Sample or pad to fnum frames
        if n >= fnum:
            step = n // fnum
            sampled = frames[::step][:fnum]
        else:
            sampled = list(frames)
            while len(sampled) < fnum:
                sampled.append(frames[-1])

        # Stack: (T, H, W, C)  — already RGB, no channel flip needed
        stacked = np.stack(sampled, axis=0).astype(np.float32)

        # (T, H, W, C) -> (T, C, H, W), normalize to [0, 1]
        tensor = torch.from_numpy(stacked).to(self.device).permute(0, 3, 1, 2) / 255.0

        # Resize to 224×224
        tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize with ImageNet stats
        # tensor shape: (T, C, H, W) → unsqueeze to (1, T, C, H, W) for broadcast
        tensor = tensor.unsqueeze(0)                      # (1, T, C, H, W)
        tensor = (tensor - self._mean) / self._std

        return tensor  # (1, T, C, H, W)

    # ---------------------------------------------------------------------- #
    # Similarity scoring
    # ---------------------------------------------------------------------- #

    def compute_scores(
        self,
        frames_list: List[List[np.ndarray]],
        texts: List[str],
        fnum: int = 8,
        temperature: float = 1.0,
        batch_size: int = 32,
    ) -> List:
        """
        Compute video-text cosine similarity for each (frames, text) pair.

        frames_list : list of frame lists (each list is one segment's frames in RGB)
        texts       : corresponding text for each segment
        Returns     : list of float scores (None for failed/empty segments)
        """
        n = len(frames_list)
        if n == 0:
            return []

        result = [None] * n

        # Filter out invalid inputs
        valid_indices, valid_frames, valid_texts = [], [], []
        for i, (frames, text) in enumerate(zip(frames_list, texts)):
            if not frames:
                continue
            valid_indices.append(i)
            valid_frames.append(frames)
            valid_texts.append(text or "")

        if not valid_frames:
            return result

        try:
            with torch.no_grad():
                # Encode all texts at once
                text_inputs = self._text_encoder.tokenize(
                    valid_texts, truncate=True
                ).to(self.device)
                text_features = self.model.text_model.encode_text(text_inputs).float()
                text_features = F.normalize(text_features, dim=-1)

                # Encode videos in batches
                for b_start in range(0, len(valid_frames), batch_size):
                    b_end = min(b_start + batch_size, len(valid_frames))
                    batch_frames = valid_frames[b_start:b_end]

                    try:
                        video_tensors = torch.cat(
                            [self.preprocess_frames(f, fnum) for f in batch_frames], dim=0
                        )  # (B, T, C, H, W)
                        video_features = self.model.vision_model.get_vid_features(
                            video_tensors
                        ).float()
                        video_features = F.normalize(video_features, dim=-1)

                        batch_text_feat = text_features[b_start:b_end]
                        scores = (video_features * batch_text_feat).sum(dim=-1) * temperature

                        for j, score in enumerate(scores):
                            result[valid_indices[b_start + j]] = score.item()

                        del video_tensors, video_features
                    except Exception as e:
                        print(f"[Warning] Batch {b_start}:{b_end} failed: {e}")

                torch.cuda.empty_cache()

        except Exception as e:
            print(f"[Error] compute_scores failed: {e}")

        return result
