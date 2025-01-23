import kbmod as kb
import numpy as np
import os

from random_selections import generate_random_selections

"""
Script for generating the false positive stamps for
kbmod-ml on hyak. Assumes that you have a number of kbmod results
run with bad angles off the ecliptic. The given paths in
this script are the ones that have been used previously,
however you can generate your own base dataset by running
a KBMOD search with the following generator_config:

```
generator_config = {
    "name": "EclipticCenteredSearch",
    "velocities": [92.0, 526.0, 257],
    "angles": [-np.pi / 4, -np.pi / 2, 128],
    "angle_units": "radian",
    "velocity_units": "pix / d",
    "given_ecliptic": None,
}
```

and then replacing the `wu_paths` and `res_paths` with your
`WorkUnit`s and result files respectively.
"""

if __name__ == "__main__":
    wu_dir = "/mmfs1/gscratch/dirac/wbeebe/kbmod/kbmod_wf/kbmod_new_ic/staging/stagingTNOs_3/results"
    wu_paths = [
        "uris_20x20_420au_2019-04-02_and_2019-05-07_slice0_patch5817_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-02_and_2019-05-07_slice1_patch5818_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-02_and_2019-05-07_slice2_patch5828_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-02_and_2019-05-07_slice4_patch5827_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice0_patch7782_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice1_patch7794_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice3_patch7793_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice4_patch7770_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice5_patch7795_lim5000.collection.wu.42.repro",
        "uris_20x20_420au_2019-04-03_and_2019-05-05_slice6_patch7781_lim5000.collection.wu.42.repro",
    ]

    res_dir = "/mmfs1/home/maxwest/dirac/zp_corr"
    res_paths = [
        "slice0/full_result.ecsv",
        "slice1/full_result.ecsv",
        "slice2/full_result.ecsv",
        "slice4/full_result.ecsv",
        "slice0_03/full_result.ecsv",
        "slice1_03/full_result.ecsv",
        "slice3_03/full_result.ecsv",
        "slice4_03/full_result.ecsv",
        "slice5_03/full_result.ecsv",
        "slice6_03/full_result.ecsv",
    ]

    for w, r in zip(wu_paths, res_paths):
        wu = kb.work_unit.WorkUnit.from_sharded_fits(w, wu_dir)
        res = kb.results.Results.read_table(os.path.join(res_dir, r))
        vis = kb.analysis.visualizer.Visualizer(wu.im_stack, res)
        vis.generate_all_stamps(radius=12)

        l = lambda i: [j.image for j in i]
        y = np.apply_along_axis(l, 1, vis.results["all_stamps"].data)

        coadds = []
        for i in range(len(y)):
            row = y[i]
            row = row[np.sum(row, axis = (-1, -2)) != 0]

            c = generate_random_selections(row, 50)

            for a in c:
                coadds.append(a)

        coadds = np.array(coadds)
        np.save(f"/mmfs1/home/maxwest/dirac/false_positive_stamps/{w}_stamps.npy", coadds)
