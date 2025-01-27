import numpy as np

from kbmod.search import (
    RawImage,
    LayeredImage,
    ImageStack,
    PSF,
    get_coadded_stamps,
    Trajectory,
    StampParameters,
    StampType
)

def generate_random_selections(rows, n, variances=None, visits=None):
    """Takes a 50x50 stamp row and generates N varied sets of stamps,
    varied by center offset, rotation, mirror, and random selection."""
    # Setup
    minimum_n = 25
    if len(rows) <= minimum_n:
        n_in_range = [len(rows)]
    else:
        n_in_range = range(minimum_n, len(rows))

    # possible warping options
    center_pix = (12, 12)
    x_offset_options = [-1, 0, 1]
    y_offset_options = [-1, 0, 1]
    mirror_axes = [None, 0, 1, 2]

    # make a randomized warping decision for
    # each n stamp that we are going to generate
    # ahead of time.
    n_in = np.random.choice(n_in_range, n)
    x_offsets = np.random.choice(x_offset_options, n)
    y_offsets = np.random.choice(y_offset_options, n)
    rots = np.random.choice(4, n)
    flips = np.random.choice(mirror_axes, n)

    # grab all the stamps for this run.
    coadds = []
    viss = []
    for i in range(n):
        inds = np.random.choice(len(rows), n_in[i])
        sub_stamps = rows[inds]

        x = center_pix[0] + x_offsets[i]
        y = center_pix[1] + y_offsets[i]

        # perform warping
        sub_stamps = sub_stamps[:,x-10:x+11,y-10:y+11]
        sub_stamps = np.flip(sub_stamps, flips[i]) if flips[i] != 0 else sub_stamps
        sub_stamps = np.rot90(sub_stamps, k=rots[i], axes=(1,2))

        # if we have the variances, get the stamp and
        # do the same warping as above.
        if variances is not None:
            sub_vars = variances[inds]
            sub_vars = sub_stamps[:,x-10:x+11,y-10:y+11]
            sub_vars = np.flip(sub_stamps, flips[i]) if flips[i] != 0 else sub_stamps
            sub_vars = np.rot90(sub_stamps, k=rots[i], axes=(1,2))
        else:
            sub_vars = [None] * len(inds)

        if visits is not None:
            sub_vis = visits[inds]

        l_imgs = []
        for s, v in zip(sub_stamps, sub_vars):
            # if variance supplied, pass that along. Otherwise create a default if it's not used.
            rivar = RawImage(v) if v is not None else RawImage(21, 21, 0.0)
            # create a layered image with science, variance, and empty mask.
            layered_img = LayeredImage(RawImage(s), rivar, RawImage(21, 21, 0.0), PSF(1.0))
            l_imgs.append(layered_img)

        im_stack = ImageStack(l_imgs)
        i = im_stack.img_count()
        params = StampParameters()
        params.radius = 10
        params.do_filtering = False
        s_types = [StampType.STAMP_MEDIAN, StampType.STAMP_MEAN, StampType.STAMP_SUM]

        stamps = [] 

        for st in s_types:
            params.stamp_type = st
            # Create a coadded stamp using a Trajectory with no motion (i.e., just the center pixels.)
            stamp = get_coadded_stamps(im_stack, [Trajectory(10, 10)], [[True]* i], params, False)
            stamps.append(stamp[0].image)
        coadds.append(stamps)
        viss.append(sub_vis)
    return coadds, viss