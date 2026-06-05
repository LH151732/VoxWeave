"""Vendored Mel-Band Roformer (frozen).

来源: lucidrains/BS-RoFormer (MIT), 经 audio-separator (uvr_lib_v5) 副本。
冻结进仓的原因: PyPI 上 bs-roformer 最新 (1.1.0) 架构已漂移 (加了 hyper-connections/pope),
社区流通的 Mel-Band Roformer ckpt (如 Kim vocals) 在新版 load_state_dict 失配 (498 missing /
120 unexpected)。这份是与那些 ckpt 对齐的版本 (实测 0 missing/0 unexpected)。改动: 无。
"""
