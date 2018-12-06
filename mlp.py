#!/usr/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def get_two_spiral_data():
	c1 = np.empty((96, 4))
	c2 = np.empty((96, 4))
	for i in range(96):
		theta = i * np.pi / 16
		r = 6.5 * (104 - i) / 104
		x = r * np.sin(theta)
		y = r * np.cos(theta)
		c1[i] = (x,y,1,0)
		c2[i] = (-x,-y,0,1)
	return np.vstack((c1, c2)).copy()

def main():
	c = np.empty((96 * 2, 4))
	c = get_two_spiral_data()
	# print(c)
	# print(c.ndim)
	# print(c.shape)
	plt.plot(c[0:95,0], c[0:95,1], 'rs', c[96:,0], c[96:,1], 'b^')
	# plt.show()

	''' initialize '''
	nvectors = c.shape[0] # row size of input pattern
	b_ninpdim = 2
	b_ninpdim_1 = b_ninpdim + 1 
	i_nhid = 40
	i_nhid_1 = i_nhid + 1
	j_nhid = 20
	j_nhid_1 = j_nhid + 1
	k_noutdim = 1

	wkj = np.random.random((k_noutdim, j_nhid_1))
	wkj_temp = np.zeros(wkj.shape)
	wji = np.random.random((j_nhid_1, i_nhid_1))
	wji_temp = np.zeros(wkj.shape)
	wib = np.random.random((i_nhid, b_ninpdim_1))

	olddelwkj = np.zeros((k_noutdim, j_nhid_1))
	print(olddelwkj.shape)
	olddelwji = np.zeros((j_nhid, i_nhid_1))
	print(olddelwji.shape)
	olddelwib = np.zeros((i_nhid, b_ninpdim_1))
	print(olddelwib.shape)


if __name__ == "__main__":
	main()
