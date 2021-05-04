import torch

#%% 1D with torch
t=torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.]) #1차원 벡터 만들기
print(t) #tensor([0., 1., 2., 3., 4., 5., 6.]) 
print(t.dim()) #rank 차원 ; 1
print(t.shape) #shape 크기 ; tensor([0., 1., 2., 3., 4., 5., 6.])
print(t.size()) #shape 크기 ; tensor([0., 1., 2., 3., 4., 5., 6.])

print(t[0], t[1], t[-1]) ; print(t[2:5], t[4:-1]) ; print(t[:2], t[3:])  
#tensor(0.) tensor(1.) tensor(6.) ; tensor([2., 3., 4.]) tensor([4., 5.]) :tensor([0., 1.]) tensor([3., 4., 5., 6.])

#%% 2D with PyTorch

t = torch.FloatTensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]) #2차원 행렬 만들기

print(t)
print('Rank of t :', t.dim()) #2차원 행렬 차원 출력 
print('shape of t :', t.size()) #2차원 행렬 크기 출력 (4,3) 4행 3열

print(t[:, 1]) # 첫번째 차원 > 전체 선택 [두번째 차원 > 첫번째] ; tensor([ 2.,  5.,  8., 11.])
print(t[1, 1]) # 첫번째 차원 > 첫번째 [두번째 차원 > 첫번째] ; tensor(5.)
print(t[:, 1].size()) #tensor([ 2.,  5.,  8., 11.])

print(t[:, -1]) # 첫번째 차원 > 전체 선택 [두번째 차원 > 마지막값만] ; tensor([ 3.,  6.,  9., 12.])
print(t[:, :-1]) # 첫번째 차원 > 전체 선택 [두번째 차원 > 마지막값 제외 모든값] ; tensor([[ 1.,  2.], ; [ 4.,  5.], ; [ 7.,  8.], ; [10., 11.]])
print(t[:, -1:]) # 첫번째 차원 > 전체 선택 [두번째 차원 > 마지막값] ; tensor([[ 3.], ; [ 6.], ;  [ 9.], ; [12.]])