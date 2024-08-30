# /usr/bin/env python3
# -*- coding: utf-8 -*-
# Matcha-GNN

import java

with java:
    from java.io import File

import mowl
mowl.init_jvm("20g")

from .training import Trainer, GridSearchCV
from .gnns import *
from .datasets import *
from .loaders import *
from .evaluators import *
from .lms import *
from .samplers import *


__author__      = "Laura Balbi"

