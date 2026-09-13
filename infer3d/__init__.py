"""Infer3D: test-time analysis-by-synthesis for out-of-distribution single-view 3D.

A frozen Splatter-Image lifter constrains the 3D output while a generative prior
(DiffAE or StyleGAN) latent + pose are optimized to match the input view, which
recovers reconstruction quality under out-of-distribution camera poses where the
feed-forward lifter alone degrades.
"""
__version__ = "0.1.0"
