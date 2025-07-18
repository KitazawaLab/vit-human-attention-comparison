# Codes of "Emergence of Human-Like Attention in Self-Supervised Vision Transformers: an eye-tracking study"

## Examples of the gaze patterns of the Vision Transformers
Five clips (no. 1, 2, 5, 8, 10) were taken from TV programmes for young
children ‘‘Okaasan to Issho’’ (NHK, Japan Broadcasting Corporation). Four (no. 4, 6, 7, 9) were taken from a film "Always: Sunset on Third Street" (Toho Co., Ltd). Clip no. 3 was taken from a film "A.I.-Artifical Intelligence" (Warner bros.).
### DINO ViT (G1; n=24)
<div><video controls src="https://github.com/user-attachments/assets/bddd4501-adff-4670-ada0-a79f981000a9" muted="false"></video></div>

### SL ViT (Top 5 heads; n=30)
<div><video controls src="https://github.com/user-attachments/assets/fbed9c38-f73a-46b5-b3bc-0c5c722c428f" muted="false"></video></div>

## Demo & Model Weights
The model weights and demo can be downloaded from the following OSF link: <https://osf.io/c3snp/>

The demo extracts features and attention maps corresponding to G1 (foreground center), G2 (entire foreground), and G3 (background) from images.
For more details, please refer to the OSF page.

### Model training codes
- Self-supervised learning: [DINO](https://github.com/facebookresearch/dino)
- Supervised learning: [DeiT](https://github.com/facebookresearch/deit/)

## Datasets
- Eye tracking dataset:
  - [Nakano et al., 2010](https://osf.io/x6f8t/)
  - [Costela and Woods 2019](https://osf.io/g64tk/)
- [Animal Parts Dataset](https://www.robots.ox.ac.uk/~vgg/data/animal_parts/)

## Analysis codes
### Model definition of vision transformers
Codes were modified from the repository of [DINO](https://github.com/facebookresearch/dino).
- `vision_transformer.py`
- `utils.py`

### Acquisition of gaze positions from ViTs
- `utils_analysis.py` : Utility functions for analysis
- `get_model_gaze_pos_N2010.ipynb`
- `get_model_official_gaze_pos_N2010.ipynb`
- `get_model_gaze_pos_CW2019.ipynb`
- `get_model_official_gaze_pos_CW2019.ipynb`
- `get_model_gaze_pos_animals.ipynb`

### Calculation of the MDS distance
- `get_MDS_distance.ipynb`: Conversion and merging of .mat files into .npz files

### Visualisation of MDS analysis results
- `visualize_mds_scale_N2010_top5.ipynb`: (Fig. 2a-b, Fig. 4a, Sup. fig. 1)
- `visualize_mds_scale_CW2019_top5.ipynb`: (Fig. 3b-c, Sup. fig. 2)
- `visualize_mds_dist_comparison_genre.ipynb`: (fig. 3d)

### Visualisation of gaze positions for examples
- `visualize_gaze_points_N2010.ipynb`: (Fig. 2c)
- `visualize_gaze_pos_CW2019.ipynb`: (Fig. 3a)

### Clustering of self-attention maps
- `attention_nbclust.ipynb`
- `get_attn_cos-sim_groups.ipynb`: (Fig. 4b-d)

### Acquisition and visualisation of attention maps for the examples
- `get_attention_groups_N2010.ipynb`
- `visualize_attention_N2010_groups.ipynb`: (Fig. 5a top)
- `get_visualize_attention_groups_CW2019.ipynb`: (Fig. 5a bottom)
- `get_attention_groups_animals.ipynb`: Get attention maps for animal images
- `visualize_attention_animals_groups_valdata.ipynb`: (Fig. 5b)
- `get_attention_groups_toy.ipynb`
- `visualize_attention_toy_groups.ipynb`: (Fig. 7)

### Viewing proportion analysis
#### Calculation of Viewing proportion
- `get_viewing_prop_goodsubj_N2010.ipynb`
- `get_viewing_prop_vit_N2010.ipynb`
- `get_viewing_prop_gbvs_N2010.ipynb`
- `get_viewing_prop_body_parts.ipynb`
- `get_viewing_prop_animal.ipynb`

#### Visualization of viewing proportion
- `visualize_viewing_prop_parts_N2010.ipynb`: (Fig. 6a)
- `visualize_gaze_points_gazew.ipynb`: (Fig. 6b)
- `visualize_gazew_animals_g1-3.ipynb`: face-viewing proportions for non-human animals and humans (Fig. 6c)
- `visualize_gaze_density_viewing_prop.ipynb`: (Fig. 6d)
- `visualize_viewing_prop_N2010.ipynb`: (Fig. 6e-f)

### Random sampling from Imagenet-1k
- `random_choice_imagenet.ipynb`: (Sup. Fig. 3)

## Reference
- Yamamoto, T., Akahoshi, H., & Kitazawa, S. (2025). **Emergence of human-like attention and distinct head clusters in self-supervised vision transformers**: A comparative eye-tracking study. _Neural Networks_, 107595, 107595.
- Yamamoto, T., Akahoshi, H., & Kitazawa, S. (2024). **Emergence of Human-Like Attention in Self-Supervised Vision Transformers: an eye-tracking study**. _arXiv_. http://arxiv.org/abs/2410.22768
