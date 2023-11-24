# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.vecr import VECR
from DAFormer.mmseg.models.uda.vecr_prog import VECR_ProG

__all__ = ['DACS', 'VECR', 'VECR_ProG']
