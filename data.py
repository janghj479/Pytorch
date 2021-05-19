# mne imports
import mne
from mne import io
from mne.datasets import sample


import torch
import torch.utils.data as data_utils 


"""
def to_one_hot(y, n_dims=None):
    
    #Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. 
    
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot
"""

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
#K.set_image_data_format('channels_last')

##################### Process, filter and epoch the data ######################
data_path = sample.data_path()


# Set parameters and read data
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
event_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
tmin, tmax = -0., 1
event_id = dict(aud_l=1, aud_r=2, vis_l=3, vis_r=4)

# Setup for reading the raw data
raw = io.Raw(raw_fname, preload=True, verbose=False)
raw.filter(2, None, method='iir')  # replace baselining with high-pass
events = mne.read_events(event_fname)

raw.info['bads'] = ['MEG 2443']  # set bad channels
picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=False,
                    picks=picks, baseline=None, preload=True, verbose=False)
labels = epochs.events[:, -1]

# extract raw data. scale by 1000 due to scaling sensitivity in deep learning
X = epochs.get_data()*1000 # format is in (trials, channels, samples)
y = labels

kernels, chans, samples = 1, 60, 151

# take 50/25/25 percent of the data to train/validate/test
X_train      = X[0:144,]
Y_train      = y[0:144]
X_validate   = X[144:216,]
Y_validate   = y[144:216]
X_test       = X[216:,]
Y_test       = y[216:]

############################# EEGNet portion ##################################
#tensor to one_hot_encoding

Y_train = torch.Tensor(Y_train).random_(0, 4) #Y_train.shape, Y_train.size
Y_train = torch.nn.functional.one_hot(Y_train.to(torch.int64), 4) #print(Y_train)

Y_validate = torch.Tensor(Y_validate).random_(0, 4)
Y_validate = torch.nn.functional.one_hot(Y_validate.to(torch.int64), 4)

Y_test = torch.Tensor(Y_test).random_(0, 4)
Y_test = torch.nn.functional.one_hot(Y_test.to(torch.int64), 4)


# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
# torch.Tensor() / tensor로 변환할 때 새 메모리를 할당한다.
# torch.from_numpy() / tensor로 변환할 때, 원래 메모리를 상속받는다. (=as_tensor())

X_train      = torch.from_numpy(X_train.reshape(X_train.shape[0], chans, samples, kernels))
X_validate   = torch.from_numpy(X_validate.reshape(X_validate.shape[0], chans, samples, kernels))
X_test       = torch.from_numpy(X_test.reshape(X_test.shape[0], chans, samples, kernels))
   

trn = data_utils.TensorDataset(X_train, Y_train)
trn_loader = data_utils.DataLoader(trn, batch_size = 16, shuffle=True)

val = data_utils.TensorDataset(X_validate, Y_validate)
val_loader = data_utils.DataLoader(val, batch_size = 16, shuffle=False)

test = data_utils.TensorDataset(X_test, Y_test)
test_loader = data_utils.DataLoader(test, batch_size = 16, shuffle=True)