#!/usr/bin/env python
# -*- coding: utf-8 -*-

from theano import tensor as T
x = T.matrix()
y = T.matrix()

a = T.vector()

b = T.dot(x, y)
c = T.dot(x, a)
