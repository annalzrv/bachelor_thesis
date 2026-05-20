"""Joint (x, lambda) functional ANOVA for parameter-conditioned neural networks.

Built on top of `anova/hdmr_net.py`'s TruncatedHDMR. The trick is that
TruncatedHDMR is already dimension-agnostic — it decomposes any function
of d variables into main + pair effects. We just need to feed it
joint (x, lambda) samples drawn from the right joint prior, then
aggregate the per-subset terms into the three canonical signatures:

    u_x(x)      = sum of spatial-only mains and spatial-spatial pairs
    u_lambda(l) = sum of parameter-only mains and parameter-parameter pairs
    u_{x,l}(x,l) = sum of cross pairs (one spatial, one parameter dim)

The mathematical claim is that under the product prior p(x) * p(lambda),
this decomposition is unique and the per-subset MLPs converge to the
true functional ANOVA terms as the network capacity grows.
"""
