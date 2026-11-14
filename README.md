# Vision–Text Fusion with Fixed Clinical Priors for Skin Lesion Diagnosis Without Paired Image–Text Annotations

Code for the vision-only baseline and the proposed fixed-text FiLM model (ConvNeXt-Tiny, ISIC 2017, 3 classes: benign nevus, melanoma, seborrheic keratosis).

| File | Model |
|---|---|
| `convnext_vision_only.py` | ImageNet-pretrained ConvNeXt-Tiny + MLP head (768-256-64-3) |
| `convnext_film.py` | Same, plus FiLM conditioning from a fixed clinical prompt (frozen all-MiniLM-L6-v2, computed once) applied to the pooled 768-d feature |

## Data
ISIC 2017 images were pre-resized to 192×256 and arranged as:

```
<data_root>/{Train,Valid,Test}/{B,M,SK}/image/*.png      (2000 / 150 / 600 images)
```
## Citation
<Mohammed A. Al-masni, " Vision–Text Fusion with Fixed Clinical Priors for Skin Lesion Diagnosis Without Paired Image–Text Annotations," Under Review, 2026>
