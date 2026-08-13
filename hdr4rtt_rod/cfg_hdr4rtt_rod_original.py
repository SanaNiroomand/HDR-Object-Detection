#!/usr/bin/env python3
"""
cfg_hdr4rtt_rod_original.py -- identical to cfg_hdr4rtt_rod.py but trained on
HDR4RTT's ORIGINAL (leaky) split.

Exists only to measure the cost of that leakage. Every other setting -- model
geometry, gamma_range, batch behaviour, learning rate, epochs, the capped
multiscale fix -- is inherited unchanged, so the only difference between this run
and the seqsafe run is which images are in train and test.

Comparison to draw, restricted to S3 (the continuous video, the only source with
frame adjacency):

    this model  on hdr4rtt_rod_original_test_S3.json   <- every test frame has a
                                                          near-duplicate in train
    seqsafe run on hdr4rtt_rod_seqsafe_test_S3.json    <- guard band, none do

S3-against-S3 keeps content constant; comparing whole splits would confound the
leakage with the fact that S3 is traffic-heavy video and S2 is bracketed stills.
"""
import importlib.util
import os

_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cfg_hdr4rtt_rod.py")
_spec = importlib.util.spec_from_file_location("_cfg_base", _BASE)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


class Exp(_base.Exp):
    def __init__(self):
        super().__init__()
        self.train_ann = "hdr4rtt_rod_original_train.json"
        self.val_ann = "hdr4rtt_rod_original_test.json"
        self.exp_name = "cfg_hdr4rtt_rod_original"
