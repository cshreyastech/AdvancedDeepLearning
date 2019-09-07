#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as T
x = T.vector()
y = T.vector()

a = x * y

d = T.dot(x, y)
