from astropy.table import Table
import os
import glob
import numpy as np

from random_selections import generate_random_selections

"""
Script for generating the true positive stamps for kbmod-ml
on hyak. Based on Steven's fake cutouts.
"""

if __name__ == "__main__":
    files = glob.glob("/gscratch/dirac/DEEP/collab/fakes_cutouts/data/*/npy")
    sub_files = files

    coadds = []
    coadd_ids = []
    orbit_ids = []
    visits = []
    i = 1
    coadd_id = -1
    for f in sub_files:
        i_path = f + "/image.npy"
        v_path = f + "/variance.npy"
        vi_path = f + "/visits.npy"

        if os.path.exists(i_path) and os.path.exists(v_path):
            image = np.load(i_path)
            var = np.load(v_path)
            vis = np.load(vi_path)
            orb_id = int(f.split("/")[-2])

            if len(image) > 0:
                c, v = generate_random_selections(image, var, vis, 10)
                for a, b in zip(c, v):
                    coadd_id += 1
                    coadds.append(a)
                    visits.append(b)
                    coadd_ids.append(coadd_id)
                    orbit_ids.append(orb_id)


        if i % 1000 == 0:
            print(f"completed chunk {i /1000}")
        i += 1
    coadds = np.array(coadds)
    np.save(f"/mmfs1/home/maxwest/dirac/true_positive_stamps_v2/tp_stamps.npy", coadds)

    # the true positive set has extra metadata.
    t = Table()
    t["coadd_id"] = coadd_ids
    t["orbit_id"] = orbit_ids
    t["vists"] = visits
    t.write("/mmfs1/home/maxwest/dirac/true_positive_stamps_v2/xmatch_meta.ecsv", format="ascii.ecsv")
