import numpy as np

#%% 1D with Numpy
t=np.array([0.,1.,2.,3.,4.,5.,6.]) #1차원 벡터 만들기

print(t)
print('Rank of t :', t.ndim) #1차원 벡터 차원 출력 ; 1차원-벡터 / 2차원-행렬 / 3차원-텐서 
print('shape of t :', t.shape) #1차원 벡터 크기 출력; (7,) >> (1,7) 의미

#%% Numpy 기초 이해

print('t[0], t[1], t[-1]: ',t[0], t[1], t[-1] ) # 인덱스로 원소 접근 
print('t[2:5] t[4:-1]  = ', t[2:5], t[4:-1]) #(시작값:끝값+1)
print('t[:2] t[3:]     = ', t[:2], t[3:]) #시작번호 생략시 처음부터

#%% 2D with Numpy
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]) #2차원 행렬 만들기

print(t)
print('Rank of t :', t.ndim) #2차원 행렬 차원 출력 
print('shape of t :', t.shape) #2차원 행렬 크기 출력 (4,3) 4행 3열




