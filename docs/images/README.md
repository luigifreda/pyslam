Diagrams are written in [Mermaid](https://mermaid.js.org/). Each `<name>.png` is rendered from the `<name>.mmd` source next to it; sources can also be edited and previewed at https://mermaid.live/edit.

- `indirect_slam/`: feature-based SLAM (`main_slam.py`) workflow, components and modules.
- `direct_slam/`: direct / semi-direct SLAM (`main_slam_direct.py`) workflow, components, pose tracker and map.
- `scene_from_views/`: 3D scene reconstruction from views.
- `shared/`: infrastructure shared by the pipelines (visual model catalog).

To regenerate the PNGs, run [`render_diagrams.py`](./render_diagrams.py). It needs Node.js (>= 18.19) and Chrome/Chromium; it uses `mmdc` from PATH or fetches the pinned [mermaid-cli](https://github.com/mermaid-js/mermaid-cli) with `npx`.

```bash
python docs/images/render_diagrams.py                 # render diagrams whose PNG is missing or older than the source
python docs/images/render_diagrams.py --all           # re-render everything
python docs/images/render_diagrams.py direct_slam     # one folder, or pass individual .mmd files
python docs/images/render_diagrams.py --list          # show which PNGs are outdated
```

Renderer regression checks run with `python -m unittest discover -s docs/images -p 'test_*.py'`.

A PNG is replaced only after rendering succeeds. A failed render preserves the previous PNG and can be retried with the same command.

Sources without a front-matter `config:` block are rendered with the hand-drawn look used on mermaid.live. Write edge labels as Mermaid markdown strings, e.g. `-->|"`_has-a_`"|`, so that they render in italics with both mermaid.live and mermaid-cli.
