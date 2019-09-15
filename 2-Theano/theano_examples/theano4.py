#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
from theano import tensor as T
#tensor3 =T.Tensortype(broadcastable=(False, False, False),dtype='float32')
#x =tensor3()
dtype='float32'

ndim=1
broadcast = (False,) * ndim
name=None

x = T.TensorType(dtype, broadcast)(name)
