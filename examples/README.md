# Examples

Scripts that show how to use parts of pySLAM. They are not the automated test suite. Run that with [`tests/run_all.sh`](../tests/run_all.sh).

Filenames start with `example_`. Pytest does not collect this folder.

Sample images used by the cv and depth scripts are in `examples/data`. Many of those scripts expect to be started from their own directory, because they load `../data/...`.

| Topic | Start here |
|-------|------------|
| Local features | [`cv/example_feature_manager.py`](cv/example_feature_manager.py) |
| Loop closing | [`loopclosing/example_loop_detector.py`](loopclosing/example_loop_detector.py) |
| Depth models | [`depth_estimation/`](depth_estimation/) |
| Semantics | [`semantics/`](semantics/) |
| Third-party features | [`thirdparty/`](thirdparty/) |
| Pangolin | [`pangolin/`](pangolin/) and [`glutils/`](glutils/) |

Vocabulary helpers for a new loop-closing descriptor are `loopclosing/example_gen_des_array_from_imgs.py`, `loopclosing/example_gen_dbow_voc_from_des_array.py`, and `loopclosing/example_gen_vlad_voc_from_des_array.py`.
