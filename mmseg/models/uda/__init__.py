# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from mmseg.models.uda.dacs import DACS
from mmseg.models.uda.vecr import VECR
from mmseg.models.uda.vecr_prog import VECR_ProG
from mmseg.models.uda.vecr_prow import VECR_ProW

__all__ = ['DACS', 'VECR', 'VECR_ProG', 'VECR_ProW']
