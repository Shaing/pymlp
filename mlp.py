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

def get_double_moon_data():
	N = 250
	theta1 = np.linspace(-180, 180, N) * np.pi / 360
	r = 8
	x1 = -5 + (r * np.sin(theta1)) + np.random.random((1, N))
	y1 = (r * np.cos(theta1)) + np.random.random((1, N))
	x2 = 5 + (r * np.sin(theta1)) + np.random.random((1, N))
	y2 = (-r * np.cos(theta1)) + np.random.random((1, N))

	l_row = []
	l_col = []
	for i in range(250):
		l_row.clear()
		l_row.append(x1[0][i])
		l_row.append(y1[0][i])
		l_row.append(1)
		l_row.append(0)
		l_col.append(l_row.copy())
	
	for i in range(250):
		l_row.clear()
		l_row.append(x2[0][i])
		l_row.append(y2[0][i])
		l_row.append(0)
		l_row.append(1)
		l_col.append(l_row.copy())
	c = np.empty((500, 4))
	for i in range(500):
		c[i] = l_col[i]
	return c.copy()

def sigmoidal(_s):
	return 1 / (1 + np.exp(-_s))

def main():
	# c = np.empty((96 * 2, 4))
	# c = get_two_spiral_data()
	# print(c)
	# print(c.ndim)
	# print(c.shape)
	# plt.plot(c[0:95,0], c[0:95,1], 'rs', c[96:,0], c[96:,1], 'b^')
	# plt.show()
	c = get_double_moon_data()
	# plt.plot(c[0:249,0], c[0:249,1], 'rs', c[250:,0], c[250:,1], 'b^')
	# plt.show()

	''' initialize '''
	nvectors = c.shape[0] # row size of input pattern
	b_ninpdim = 2
	b_ninpdim_1 = b_ninpdim + 1 
	i_nhid = 10
	i_nhid_1 = i_nhid + 1
	j_nhid = 10
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

	ob = np.zeros((b_ninpdim_1, 1))
	ob[-1] = 1
	si = np.zeros((i_nhid, 1))
	oi = np.zeros((i_nhid_1, 1))
	oi[-1] = 1
	sj = np.zeros((j_nhid, 1))
	oj = np.zeros((j_nhid_1, 1))
	oj[-1] = 1
	sk = np.zeros((k_noutdim, 1))
	ok = np.zeros(sk.shape)
	dk = np.zeros((k_noutdim, 1))

	lower_limit = 0.001
	iter_max = 3000
	eta = 0.3
	beta = 0.5

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
	ite = []
	error_r = []
	while error_avg > lower_limit and _iter < iter_max:
		_iter = _iter + 1
		error = 0

		''' Forward Computation '''
		for ivector in range(nvectors):
			ob = np.array([[c[ivector][1]], [c[ivector][2]], [1]])
			dk = np.array([[c[ivector][3]]])

			for i in range(i_nhid):
				si[i] = np.dot(wib[i], ob) 
				# oi[i] = 1 / (1 + np.exp(-si[i]))
				oi[i] = sigmoidal(si[i])
			oi[-1] = 1.0
			# print(oi)

			# si = np.dot(wib, ob)
			# oi[:-1] = sigmoidal(si)
			# oi[-1] = 1.0

			for j in range(j_nhid):
				sj[j] = np.dot(wji[j], oi)
				oj[j] = sigmoidal(sj[j])
			oj[-1] = 1.0
			# sj = np.dot(wji, oi)
			# oj[:-1] = sigmoidal(sj)
			# oj[-1] = 1.0

			for k in range(k_noutdim):
				sk[k] = np.dot(wkj[k], oj)
				ok[k] = sigmoidal(sk[k])
			# sk = np.dot(wkj, oj)
			# ok = sigmoidal(sk)
			
			error = error + sum(abs(dk - ok))

			# print(si)
			# print(oi)
			# print(sj)
			# print(oj)
			# print(sk)
			# print(ok)
			# print(error)

			''' Backward learning '''
			# for k in range(k_noutdim):
			# 	delta_k[k] = (dk[k] - ok[k]) * ok[k] * (1.0 - ok[k])
			delta_k = (dk - ok) * ok * (1.0 - ok)
			
			for j in range(j_nhid_1):
				for k in range(k_noutdim):
					wkj_temp[k][j] = wkj[k][j] + \
									(eta * delta_k[0][k] * oj[j]) + \
									(beta * old_delwkj[k][j])
					old_delwkj[k][j] = (eta * delta_k[0][k] * oj[j]) + \
									(beta * old_delwkj[k][j])

			for j in range(j_nhid):
				sum_back_kj[0][j] = 0.0
				for k in range(k_noutdim):
					sum_back_kj[0][j] = sum_back_kj[0][j] + \
										(delta_k[0][k] * wkj[k][j])
				delta_j[0][j] = oj[j] * (1.0 - oj[j]) * sum_back_kj[0][j]
			# delta_j = oj[:-1] * (1.0 - oj[:-1]) * sum_back_kj

			for i in range(i_nhid_1):
				for j in range(j_nhid):
					wji_temp[j][i] = wji[j][i] + \
									(eta * delta_j[0][j] * oi[i]) + \
									(beta * old_delwji[j][i])
					old_delwji[j][i] = (eta * delta_j[0][j] * oi[i]) + \
									(beta * old_delwji[j][i])
			
			for i in range(i_nhid):
				sum_back_ji[0][i] = 0.0
				for j in range(j_nhid):
					sum_back_ji[0][i] = sum_back_ji[0][i] + \
										(delta_j[0][j] * wji[j][i])
				delta_i[0][i] = oi[i] * (1.0 - oi[i]) * sum_back_ji[0][i]
			# delta_i = oi[:-1] * (1.0 - oi[:-1]) * sum_back_ji

			for b in range(b_ninpdim_1):
				for i in range(i_nhid):
					wib[i][b] = wib[i][b] + \
								(eta * delta_i[0][i] * ob[b]) + \
								(beta * old_delwib[i][b])
					old_delwib[i][b] = (eta * delta_i[0][i] * ob[b]) + \
								(beta * old_delwib[i][b])

			wkj = wkj_temp.copy()
			wji = wji_temp.copy()
		
		ite.append(_iter)
		error_avg = error / nvectors
		error_r.append(error_avg)
		print('[iter] {} [error_avg] {}'.format(_iter, error_avg))
	plt.clf()
	plt.figure(num=1)
	plt.plot(ite, error_r)
	# plt.show()

	plt.figure(num=2)
	plt.plot(c[0:249,0], c[0:249,1], 'rs', c[250:,0], c[250:,1], 'b^')
	for ix in range(-15, 15):
		for iy in range(-15, 15):
			ob = np.array([[ix], [iy], [1]]) 
				
			for i in range(i_nhid):
				si[i] = np.dot(wib[i], ob) 
				oi[i] = sigmoidal(si[i])
			oi[-1] = 1.0

			for j in range(j_nhid):
				sj[j] = np.dot(wji[j], oi)
				oj[j] = sigmoidal(sj[j])
			oj[-1] = 1.0

			for k in range(k_noutdim):
				sk[k] = np.dot(wkj[k], oj)
				ok[k] = sigmoidal(sk[k])

			if ok[0][0] > 0.5:
				plt.plot(ob[0][0], ob[1][0], 'yo')
			elif ok[0][0] < 0.5:
				plt.plot(ob[0][0], ob[1][0], 'k*')
	plt.show()

if __name__ == "__main__":
	main()
