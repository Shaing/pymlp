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

def sigmoidal(_s):
	return 1 / (1 + np.exp(-_s))

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
	wji = np.random.random((j_nhid, i_nhid_1))
	wji_temp = np.zeros(wji.shape)
	wib = np.random.random((i_nhid, b_ninpdim_1))

	old_delwkj = np.zeros((k_noutdim, j_nhid_1))
	old_delwji = np.zeros((j_nhid, i_nhid_1))
	old_delwib = np.zeros((i_nhid, b_ninpdim_1))

	ob = np.zeros((1, b_ninpdim_1))
	ob[0][-1] = 1
	si = np.zeros((1, i_nhid))
	oi = np.zeros((1, i_nhid_1))
	oi[0][-1] = 1
	sj = np.zeros((1, j_nhid))
	oj = np.zeros((1, j_nhid_1))
	oj[0][-1] = 1
	sk = np.zeros((1, k_noutdim))
	ok = np.zeros(sk.shape)
	dk = np.zeros((1, 1))

	lower_limit = 0.001
	iter_max = 15000
	eta = 0.1
	beta = 0.3

	_iter = 0
	iter_loop = 0
	error_avg = 10

	''' internal variables '''
	delta_k = np.zeros((1, k_noutdim))
	sum_back_kj = np.zeros((1, j_nhid))
	delta_j = np.zeros((1, j_nhid))
	sum_back_ji = np.zeros((1, i_nhid))
	delta_i = np.zeros((1, i_nhid))

	''' start F. & B. '''

	while error_avg > lower_limit and _iter < iter_max:
		_iter = _iter + 1
		error = 0

		''' Forward Computation '''
		for ivector in range(nvectors):
			ob[0] = [c[ivector][1], c[ivector][2], 1]
			dk[0] = [c[ivector][3]]

			for i in range(i_nhid):
				si[0][i] = np.vdot(wib[i], ob) 
				# oi[0][i] = 1 / (1 + np.exp(-si[0][i]))
				oi[0][i] = sigmoidal(si[0][i])
			oi[0][-1] = 1.0

			for j in range(j_nhid):
				sj[0][j] = np.vdot(wji[j], oi)
				oj[0][j] = sigmoidal(sj[0][j])
			oj[0][-1] = 1.0

			for k in range(k_noutdim):
				sk[0][k] = np.vdot(wkj[k], oj)
				ok[0][k] = sigmoidal(sk[0][k])
			
			error = error + sum(abs(dk - ok))

			# print(si)
			# print(oi)
			# print(sj)
			# print(oj)
			# print(sk)
			# print(ok)
			# print(error)

			''' Backward learning '''
			for k in range(k_noutdim):
				delta_k[0][k] = (dk[0][k] - ok[0][k]) * ok[0][k] * (1.0 - ok[0][k])
			
			for j in range(j_nhid_1):
				for k in range(k_noutdim):
					wkj_temp[k][j] = wkj[k][j] + \
									(eta * delta_k[0][k] * oj[0][j]) + \
									(beta * old_delwkj[k][j])
					old_delwkj[k][j] = (eta * delta_k[0][k] * oj[0][j]) + \
									(beta * old_delwkj[k][j])
			
			for j in range(j_nhid):
				sum_back_kj[0][j] = 0.0
				for k in range(k_noutdim):
					sum_back_kj[0][j] = sum_back_kj[0][j] + \
										(delta_k[0][k] * wkj[k][j])
				delta_j[0][j] = oj[0][j] * (1.0 - oj[0][j]) * sum_back_kj[0][j]

			for i in range(i_nhid_1):
				for j in range(j_nhid):
					wji_temp[j][i] = wji[j][i] + \
									(eta * delta_j[0][j] * oi[0][i]) + \
									(beta * old_delwji[j][i])
					old_delwji[j][i] = (eta * delta_j[0][j] * oi[0][i]) + \
									(beta * old_delwji[j][i])
			
			for i in range(i_nhid):
				sum_back_ji[0][i] = 0.0
				for j in range(j_nhid):
					sum_back_ji[0][i] = sum_back_ji[0][i] + \
										(delta_j[0][j] * wji[j][i])
				delta_i[0][i] = oi[0][i] * (1.0 - oi[0][i]) * sum_back_ji[0][i]

			for b in range(b_ninpdim_1):
				for i in range(i_nhid):
					wib[i][b] = wib[i][b] + \
								(eta * delta_i[0][i] * ob[0][b]) + \
								(beta * old_delwib[i][b])
					old_delwib[i][b] = (eta * delta_i[0][i] * ob[0][b]) + \
								(beta * old_delwib[i][b])

			wkj = wkj_temp.copy()
			wji = wji_temp.copy()
		
		# iter_loop = iter_loop + 1
		# if iter_loop == 1000:
			# print('[iter] {}'.format(_iter))
			# print('[error_avg] {}'.format(error/nvectors))
			# iter_loop = 0
		print('[iter] {}'.format(_iter))
		print('[error_avg] {}'.format(error/nvectors))

if __name__ == "__main__":
	main()
