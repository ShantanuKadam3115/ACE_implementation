import torch #type: ignore
from Phase4.backbone import TruncatedResNetBackbone, normalizedImage
from Phase3.dataset import CustomDataset

datasetPath = "Phase3\\dataset\\seq-01\\seq-01"
stride = 4

data = torch.load("seq_1_buffer.pt")

# for k, v in data.items():
#     print(k, v.shape)

random_samples = [6, 5487, 125978, 50, 98754]

dataset = CustomDataset(datasetPath)
backbone = TruncatedResNetBackbone()

for sample_idx in random_samples:
    cached_feature = data["features"][sample_idx]
    u, v = data["pixel_coords"][sample_idx]
    frame_idx = data["frame_indices"][sample_idx].item()

    
    col = (u - stride / 2) / stride
    row = (v - stride / 2) / stride

    # sanity check: mapping is exact, so this should be a whole number
    assert torch.isclose(col, col.round()), f"col not whole: {col}"
    assert torch.isclose(row, row.round()), f"row not whole: {row}"
    row, col = int(row.round().item()), int(col.round().item())

    img, _ = dataset[frame_idx]
    img_normalized = normalizedImage(img)
    with torch.no_grad():
        fresh_feature_map = backbone(img_normalized).squeeze(0)  

    fresh_feature = fresh_feature_map[:, row, col]

    is_close = torch.allclose(fresh_feature, cached_feature)
    is_equal = torch.equal(fresh_feature, cached_feature)
    max_diff = (fresh_feature - cached_feature).abs().max().item()

    print(
        f"sample {sample_idx}: frame={frame_idx}, (u,v)=({u.item()},{v.item()}) -> "
        f"(row,col)=({row},{col}) | allclose={is_close} equal={is_equal} max_diff={max_diff:.8f}"
    )
