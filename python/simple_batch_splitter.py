#!/usr/bin/env python
from mvbb.box_db import MVBBLoader
import multiprocessing, subprocess
from multiprocessing import Pool
import sys

from plugins import soft_hand

def grasp_boxes(filename):
    subprocess.call(['python', './grasp_boxes_batch.py', filename])


if __name__ == '__main__':
    try:
        import os.path
        filename = os.path.splitext(sys.argv[1])[0]
    except:
        filename = 'box_db'

    if not os.path.isfile(filename+'.csv'):
        print "Error: file", filename, "doesn't exist"
        exit()

    try:
        n_dofs = int(sys.argv[2])
        n_l = int(sys.argv[3])
    except:
        n_dofs = soft_hand.numJoints
        n_l = len(soft_hand.links_to_check)

    # for SoftHand
    box_db = MVBBLoader(filename, n_dofs, n_l)
    filenames = box_db.split_db()

    p = Pool(multiprocessing.cpu_count())
    p.map(grasp_boxes, filenames)

    box_db.join_results(filenames)
