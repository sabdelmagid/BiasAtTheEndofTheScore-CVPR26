# Bias at the End of the Score

Official code release for the paper **“Bias at the End of the Score”**, CVPR 2026.

This repository is organized into three parts:

- **Part 1 — Hypersexualization:** measuring disparate hypersexualization under reward-guided optimization.
- **Part 2 — Demographic Drift:** measuring changes in perceived demographic attributes under reward-guided optimization.
- **Part 3 — Ranking:** analyzing demographic disparities in reward-model rankings. *(Coming soon.)*

Detailed instructions for reproducing each experiment are provided in the corresponding subdirectory.

## License

Released under the [MIT License](LICENSE).

## Acknowledgements

This code builds on [ReNO](https://github.com/ExplainableML/ReNO) (MIT License, © 2024 EML), whose optimization pipeline, model wrappers, and reward implementations we adapt. If you use this code, please also cite:

```bibtex
@article{eyring2024reno,
  title   = {ReNO: Enhancing One-step Text-to-Image Models through Reward-based Noise Optimization},
  author  = {Luca Eyring and Shyamgopal Karthik and Karsten Roth and Alexey Dosovitskiy and Zeynep Akata},
  journal = {Neural Information Processing Systems (NeurIPS)},
  year    = {2024}
}
```

We also use the following third-party models and data, each under its own license and terms of use:

- **Generators:** SDXL-Turbo, SD-Turbo (Stability AI), PixArt-α DMD, FLUX.1 (Black Forest Labs); `RewardFlux.py` is adapted from 🤗 Diffusers (Apache 2.0).
- **Reward models:** PickScore, ImageReward, HPSv2, the LAION improved aesthetic predictor, and CLIP (OpenAI / OpenCLIP).
- **Measurement models:** Freepik `nsfw_image_detector`, MediaPipe selfie multiclass segmentation, and CLIP ViT-L/14.
- **Data:** the Chicago Face Database (not redistributed; obtain it from its official source).

## Citation

If you use this code or build on this work, please cite:

*Please note that my last name is “Abdel Magid” (with a space).*

```bibtex
@inproceedings{abdelmagid2026bias,
  title     = {Bias at the End of the Score},
  author = {Salma {Abdel Magid} and Grace Guo and Esin Tureci and Amaya Dharmasiri and Vikram V. Ramaswamy and Hanspeter Pfister and Olga Russakovsky},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

Please feel free to reach out to me with any questions at **sabdelmagid@princeton.edu** 
