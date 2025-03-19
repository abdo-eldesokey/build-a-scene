<p align="center">
<img src="https://github.com/abdo-eldesokey/build-a-scene/blob/gh-page/static/images/favicon.ico" alt="method" width="100">
</p>

# Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation (ICLR 2025)
<a href='https://abdo-eldesokey.github.io/build-a-scene/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='https://arxiv.org/abs/2408.14819'><img src='https://img.shields.io/badge/ArXiv-2408.14819-red'></a> 


Official repository for "Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation"
<p align="center">
<img src="https://github.com/abdo-eldesokey/build-a-scene/blob/gh-page/static/images/teaser.jpg" alt="method" width=900>
</p>


## Getting Started
- To set up the environment and install dependencies, create a conda environemnt:
```bash
conda env create -f environment.yml
```

- Start with one of the provided notebooks `example_indoor.ipynb` or `example_outdoor.ipynb`.

## Evaluation

1) The evaluation set that was used in the paper is provided in `eval/evaluation_set`, but to generate your own set, we also provided `eval/generate_layouts_for_eval.ipynb` notebook for this purpose.

2) To run the evaluation for multi-stage object generation, run the following command from the project parent directory:
```bash
python -m eval.evaluate_generation
```

3) To run the evaluation for consistent 3D translation, run the following command from the project parent directory:
```bash
python -m eval.evaluate_consistency
```

4) To compute the average scores, use `eval/compute_metrics_avg.ipynb`.

## Future Directions
We provide a list of potential directions that can help improve this work
- The quality of the base SD1.5 is limited; an SDXL variation can significantly boost the visual quality.
- This work covered Consistent 3D Translation, but extending it to consistent 3D rotation as well can enhance the user controllability.
- Decrease the sensitivity of the model to the size and aspect ratio of the 3D box.

## Citation
If you use this code base or compare against it, please cite our paper
```
@inproceedings{
eldesokey2025buildascene,
title={Build-A-Scene: Interactive 3D Layout Control for Diffusion-Based Image Generation},
author={Abdelrahman Eldesokey and Peter Wonka},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=gg6dPtdC1C}
}

```
