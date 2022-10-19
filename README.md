# TODO
 - Single ellipse, changes w.r.t. angle of ellipse
 - Distribute ellipses random on circle
 - error due to stuck in minima or discretization --> Check loss of analytical vs. learned
 
 - Normalize analytic filter_params: divide scalar product of learned and divide analytic by norm squared analytic
 - Radon of circle with different resolutions (varying rotation invariance?)

 - Compare to ramp

 - other sampling (torch_radon)
 - check loss of optimal kernel?
 - matching of analytical svd and learned svd
 - Resolution rescaling without new learning (Both basic (0))
 - --> SVD with different scales very different?

 - BUG: Duplicated checkpoints
 - BUG: Multiruns overwrite logs