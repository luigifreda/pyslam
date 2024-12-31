# Differential Gaussian Rasterization with Camera Pose Jacobians

This software is used as the rasterization engine in the paper ["Gaussian Splatting SLAM"](https://arxiv.org/abs/2312.06741), and supports:

* Analytical gradient for SE(3) camera poses.
* Analytical gradient for rendered depth.

The code is built on top of the original [Differential Gaussian Rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) used in "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields".

If you can make use of it in your own research, please be so kind to cite both papers.


<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
</code></pre>
    <pre><code>@inproceedings{Matsuki:Murai:etal:CVPR2024,
  title={{G}aussian {S}platting {SLAM}},
  author={Hidenobu Matsuki and Riku Murai and Paul H. J. Kelly and Andrew J. Davison},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}</code></pre>

</div>
</section>

