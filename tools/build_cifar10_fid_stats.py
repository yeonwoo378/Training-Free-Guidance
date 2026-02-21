# tools/build_cifar10_fid_stats.py
import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ---- allow importing TFG modules when running from repo root ----
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluations.utils.inception import InceptionV3  # TFG FID inception


class CIFAR10TensorDataset(Dataset):
    """
    CIFAR-10을 (이미지 텐서)로 반환:
      - PIL RGB
      - (선택) 32x32 resize (TFG tasks/utils.py에서 하던 resize와 동일한 효과, 사실상 no-op)
      - ToTensor() -> float [0,1]
    이후 InceptionV3 내부에서 299 resize + (-1,1) 스케일링이 적용됨. :contentReference[oaicite:3]{index=3}
    """
    def __init__(
        self,
        root: str,
        split: str,              # "train" or "test"
        download: bool,
        targets: Optional[List[int]],  # None이면 전체, 아니면 해당 라벨만
        resize: int = 32,
    ):
        assert split in ["train", "test"]
        self.base = CIFAR10(root=root, train=(split == "train"), download=download)
        self.resize = resize
        self.to_tensor = T.ToTensor()

        if targets is None:
            self.indices = list(range(len(self.base)))
        else:
            targets_set = set(int(t) for t in targets)
            self.indices = [i for i, y in enumerate(self.base.targets) if int(y) in targets_set]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img, _ = self.base[self.indices[idx]]  # img is PIL.Image
        img = img.convert("RGB")
        if self.resize is not None:
            img = img.resize((self.resize, self.resize))  # default BICUBIC for RGB in Pillow
        x = self.to_tensor(img)  # [0,1], shape [3,H,W]
        return x


@torch.no_grad()
def get_activations_from_dataset(
    dataset: Dataset,
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    pred_arr = np.empty((len(dataset), dims), dtype=np.float64)
    start = 0

    for batch in tqdm(loader, desc="Extracting Inception features"):
        batch = batch.to(device, non_blocking=True)
        pred = model(batch)[0]  # selected block output

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()  # (B, dims)
        pred_arr[start:start + pred.shape[0]] = pred
        start += pred.shape[0]

    return pred_arr


def compute_mu_sigma(act: np.ndarray):
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def parse_targets(s: str) -> Optional[List[int]]:
    s = s.strip().lower()
    if s in ["all", "-1", "none", ""]:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    targets = [int(p) for p in parts]
    for t in targets:
        if t < 0 or t > 9:
            raise ValueError(f"CIFAR-10 class id must be in [0..9], got {t}")
    return targets


def make_out_path(base_out: Path, tag: str, multi: bool) -> Path:
    """
    - targets가 1개(or all)이면 base_out 그대로 사용 가능
    - targets가 여러 개면:
        * base_out이 .pt면: stem-{tag}.pt 로 저장
        * base_out이 디렉토리(또는 확장자 없음)이면: base_out/cifar10-fid-stats-{tag}.pt
    """
    if not multi:
        # single target: if user passed a directory, create default file name inside
        if base_out.suffix == ".pt":
            return base_out
        else:
            base_out.mkdir(parents=True, exist_ok=True)
            return base_out / f"cifar10-fid-stats-{tag}.pt"

    # multi targets
    if base_out.suffix == ".pt":
        return base_out.with_name(f"{base_out.stem}-{tag}{base_out.suffix}")
    else:
        base_out.mkdir(parents=True, exist_ok=True)
        return base_out / f"cifar10-fid-stats-{tag}.pt"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar_root", type=str, required=True,
                        help="CIFAR-10 저장/다운로드 루트 (torchvision CIFAR10 root)")
    parser.add_argument("--download", action="store_true",
                        help="데이터가 없으면 다운로드 시도")
    parser.add_argument("--split", type=str, default="both", choices=["train", "test", "both"],
                        help="real stats를 만들 때 사용할 split (TFG-like: both)")
    parser.add_argument("--targets", type=str, default="all",
                        help='예: "all" 또는 "3" 또는 "0,1,2,3" (클래스별 cache 만들기)')
    parser.add_argument("--out", type=str, required=True,
                        help="출력 경로(.pt 또는 디렉토리). targets가 여러 개면 자동으로 -{class}.pt로 저장")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="디버그용 샘플 제한. -1이면 전체 사용(권장)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dims", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    targets = parse_targets(args.targets)

    # 어떤 target들에 대해 파일을 만들지 결정
    if targets is None:
        target_list = [None]            # all
        tag_list = ["all"]
    else:
        target_list = [[t] for t in targets]
        tag_list = [str(t) for t in targets]

    multi = len(target_list) > 1
    base_out = Path(args.out)

    # FID inception (TFG)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    # (옵션) subset sampling: 매번 동일하게 뽑고 싶으면 deterministic shuffle
    g = torch.Generator()
    g.manual_seed(args.seed)

    for cur_targets, tag in zip(target_list, tag_list):
        print("=" * 80)
        print(f"[Target] {tag}  (targets={cur_targets})")
        print(f"[Split ] {args.split}")
        print(f"[Out   ] {base_out}")

        # build dataset(s)
        ds_list = []
        if args.split in ["train", "both"]:
            ds_list.append(CIFAR10TensorDataset(
                root=args.cifar_root, split="train", download=args.download,
                targets=cur_targets, resize=32
            ))
        if args.split in ["test", "both"]:
            ds_list.append(CIFAR10TensorDataset(
                root=args.cifar_root, split="test", download=args.download,
                targets=cur_targets, resize=32
            ))

        dataset = ds_list[0] if len(ds_list) == 1 else ConcatDataset(ds_list)
        n = len(dataset)
        print(f"[Data  ] #images = {n}")

        if args.num_samples is not None and args.num_samples > 0 and args.num_samples < n:
            # ConcatDataset/일반 Dataset 모두 지원하도록 인덱스 샘플링
            perm = torch.randperm(n, generator=g).tolist()[:args.num_samples]
            dataset = torch.utils.data.Subset(dataset, perm)
            print(f"[Data  ] using subset: {len(dataset)} images")

        # compute activations -> mu/sigma
        act = get_activations_from_dataset(
            dataset=dataset,
            model=model,
            batch_size=args.batch_size,
            dims=args.dims,
            device=device,
            num_workers=args.num_workers,
        )
        mu, sigma = compute_mu_sigma(act)
        print(f"[Stats ] mu.shape={mu.shape}, sigma.shape={sigma.shape}")

        out_path = make_out_path(base_out, tag=tag, multi=multi)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save((mu, sigma), str(out_path))
        print(f"[Save  ] {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
