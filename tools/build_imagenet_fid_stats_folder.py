# tools/build_imagenet_fid_stats_folder.py
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as TF
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# --- make imports work when running from repo root ---
# (tools/ 아래에 두면 repo root가 parents[1])
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from evaluations.utils.inception import InceptionV3  # TFG의 FID inception


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG", ".JPG", ".JPEG".lower())


class ImagePathDataset(Dataset):
    """
    ImageNet 폴더에서 path로 이미지를 열어:
      - RGB 변환
      - 256x256 resize (TFG tasks/utils.py가 하던 것과 동일하게 '그냥 resize' => 기본 BICUBIC)
      - ToTensor() => [0,1]
    까지 수행.
    이후 InceptionV3 내부에서 299 resize + [-1,1] normalize가 적용됩니다(TFG 그대로).
    """
    def __init__(self, paths: List[Path], resize: int = 256):
        self.paths = paths
        self.resize = resize
        self.to_tensor = TF.ToTensor()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        p = self.paths[idx]
        # PIL open
        with Image.open(p) as img:
            img = img.convert("RGB")
            if self.resize is not None:
                img = img.resize((self.resize, self.resize))  # default: BICUBIC (PIL)
            x = self.to_tensor(img)  # float32 [0,1], shape [3,H,W]
        return x


def list_wnids(split_dir: Path) -> List[str]:
    wnids = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    if len(wnids) == 0:
        raise RuntimeError(f"No class folders found under: {split_dir}")
    return wnids


def resolve_target_to_wnid(split_dir: Path, target: str) -> Tuple[str, Union[int, None]]:
    """
    target:
      - '111' 같은 숫자면: split_dir 하위 폴더 이름(wnid)을 정렬한 뒤 index로 선택
      - 'n01930112' 같은 wnid면: 그대로 사용
    """
    target = target.strip()
    if target.startswith("n") and len(target) == 9 and target[1:].isdigit():
        return target, None

    # numeric class index
    idx = int(target)
    wnids = list_wnids(split_dir)
    if not (0 <= idx < len(wnids)):
        raise ValueError(f"class index out of range: {idx} (found {len(wnids)} wnid folders)")
    wnid = wnids[idx]
    return wnid, idx


def gather_image_paths(class_dir: Path) -> List[Path]:
    if not class_dir.exists():
        raise FileNotFoundError(f"class directory not found: {class_dir}")

    paths = []
    for p in class_dir.rglob("*"):
        if p.is_file() and p.suffix in IMG_EXTS:
            paths.append(p)

    # 일부 환경에서 suffix 대소문자가 섞일 수 있어 추가로 보강
    if len(paths) == 0:
        # 느리지만 fallback: 확장자 검사 대신 파일 열어보기 식으로 구현할 수도 있으나
        # 여기서는 에러로 명확히 알립니다.
        raise RuntimeError(f"No image files found under: {class_dir}")
    return sorted(paths)


@torch.no_grad()
def get_activations_from_paths(
    paths: List[Path],
    model: torch.nn.Module,
    batch_size: int,
    dims: int,
    device: torch.device,
    num_workers: int,
) -> np.ndarray:
    dataset = ImagePathDataset(paths, resize=256)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    pred_arr = np.empty((len(paths), dims), dtype=np.float64)
    start = 0

    for batch in tqdm(loader, desc="Extracting Inception features"):
        batch = batch.to(device, non_blocking=True)

        pred = model(batch)[0]  # TFG fid.py와 동일하게 첫 블록 출력 사용
        # dims=2048이면 이미 (B,2048,1,1) 이지만 안전하게 처리
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()  # (B,2048)
        pred_arr[start:start + pred.shape[0]] = pred
        start += pred.shape[0]

    return pred_arr


def compute_mu_sigma(act: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet_root", type=str, required=True,
                        help="ImageNet root dir (contains train/ and maybe val/)")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "validation"],
                        help="which split folder to use")
    parser.add_argument("--target", type=str, default="111",
                        help="ImageNet class id as integer (e.g., 111) OR wnid (e.g., n01930112)")
    parser.add_argument("--out", type=str, required=True, help="output .pt path to save (mu,sigma)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dims", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device)
    split = "val" if args.split == "validation" else args.split

    split_dir = Path(args.imagenet_root) / split
    if not split_dir.exists():
        raise FileNotFoundError(f"split directory not found: {split_dir}")

    wnid, idx = resolve_target_to_wnid(split_dir, args.target)
    class_dir = split_dir / wnid

    print(f"[1] ImageNet root : {args.imagenet_root}")
    print(f"[1] Split        : {split}")
    if idx is not None:
        print(f"[1] Target index : {idx}  (folder-order mapping)")
    print(f"[1] Target wnid  : {wnid}")
    print(f"[1] Class dir    : {class_dir}")

    # (선택) index로 들어왔으면 주변 wnid도 보여줘서 sanity check 가능
    if idx is not None:
        wnids = list_wnids(split_dir)
        lo = max(0, idx - 2)
        hi = min(len(wnids), idx + 3)
        print("[1] Neighbor wnids (by sorted folder name):")
        for i in range(lo, hi):
            mark = " <==" if i == idx else ""
            print(f"    {i:4d}: {wnids[i]}{mark}")

    print("[2] Gathering image paths...")
    paths = gather_image_paths(class_dir)
    print(f"    #images = {len(paths)}")

    print("[3] Building TFG FID InceptionV3...")
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[args.dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()

    print("[4] Extracting features...")
    act = get_activations_from_paths(
        paths=paths,
        model=model,
        batch_size=args.batch_size,
        dims=args.dims,
        device=device,
        num_workers=args.num_workers,
    )

    print("[5] Computing mu/sigma...")
    mu, sigma = compute_mu_sigma(act)
    print(f"    mu.shape={mu.shape}, sigma.shape={sigma.shape}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("[6] Saving stats...")
    torch.save((mu, sigma), str(out_path))
    print(f"    saved to: {out_path}")


if __name__ == "__main__":
    main()
