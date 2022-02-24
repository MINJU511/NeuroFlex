import numpy as np
import matplotlib.pyplot as plt
import itertools

# simultae sensors
nsens = 275
corr_topo = np.linspace(1,.1,nsens)
inc_topo = corr_topo
#inc_topo  = np.flip(corr_topo)

plt.plot(corr_topo)
plt.plot(inc_topo)

# Simulate signals
tp = 300 # how many time points?
total_dur = 1 # total duration of whole signal in s
#sur = .2 # total duration of "signal part" (ERP)
TS  = 300 # sampling freq, calculate this from tp and total_dur
#sf = 5 # frequency of sinusoid component for ERP
time_vec = np.linspace(1/TS,1,TS)# time vector

'''# now the actual values
sig_corr = np.sin(2*np.pi*sf*time_vec)
# if you want to zero out some of the signal
from_which_time = .5
zero_times = time_vec > from_which_time

sig_corr[zero_times] = 0
sig_inc  = np.zeros(len(sig_corr))'''

'''# Get Averaged ERF values from the data (all channels) --------------------------
correct_erf = np.genfromtxt('correct_erf.csv', delimiter=',')
incorrect_erf = np.genfromtxt('incorrect_erf.csv', delimiter=',')
correct_erf = np.delete(correct_erf,-1,axis=1) #Remove the last column
incorrect_erf = np.delete(incorrect_erf,-1,axis=1)

correct_erf.shape #(275,300)
incorrect_erf.shape #(275,300)
sig_corr = correct_erf.mean(axis=0)
sig_inc  = np.zeros(len(sig_corr))

def mk_sample_realerf(n,m,noise_scal):
    full_corr = []
    full_inc = []
    corr = []
    inc = []
    noise_scal = noise_scal
    for i in range(n):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_corr.append(sig_corr + N)
        corr.append(np.transpose(np.asmatrix(corr_topo)) @ np.asmatrix(full_corr[i]))
    for j in range(m):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_inc.append(sig_inc + N)
        inc.append(np.transpose(np.asmatrix(inc_topo)) @ np.asmatrix(full_inc[j]))
    return corr, inc

correct, incorrect = mk_sample_realerf(50,50,0.1)
correct_array = np.array(correct) # (50, 275, 300) N of trials, channels, timepoints
incorrect_array = np.array(incorrect)  # (50, 275, 300)
'''


# Get Averaged ERF values from the data (TOP30 channels) --------------------------
correct_erf = np.genfromtxt('TOP30_correct_erf.csv', delimiter=',')
incorrect_erf = np.genfromtxt('TOP30_incorrect_erf.csv', delimiter=',')
correct_erf = np.delete(correct_erf,-1,axis=1) #Remove the last column
incorrect_erf = np.delete(incorrect_erf,-1,axis=1)

correct_erf.shape #(275,300)
incorrect_erf.shape #(275,300)
sig_corr = correct_erf.mean(axis=0)
sig_inc  = np.zeros(len(sig_corr))

def mk_sample_realerf(n,m,noise_scal):
    full_corr = []
    full_inc = []
    corr = []
    inc = []
    noise_scal = noise_scal
    for i in range(n):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_corr.append(sig_corr + N)
        corr.append(np.transpose(np.asmatrix(corr_topo)) @ np.asmatrix(full_corr[i]))
    for j in range(m):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_inc.append(sig_inc + N)
        inc.append(np.transpose(np.asmatrix(inc_topo)) @ np.asmatrix(full_inc[j]))
    return corr, inc

correct, incorrect = mk_sample_realerf(50,50,1e-15)
correct_array = np.array(correct) # (50, 275, 300) N of trials, channels, timepoints
incorrect_array = np.array(incorrect)  # (50, 275, 300)

plt.plot(time_vec,sig_corr)
plt.plot(time_vec,sig_inc)
plt.title('correct vs incorrect ERF (without noise)')

plt.plot(time_vec, correct_array[0].mean(axis=0))
plt.plot(time_vec, incorrect_array[0].mean(axis=0))
plt.title('correct vs incorrect ERF (with noise scale 1e-15)')
'''
# Generate n correct samples and m incorrect samples
def mk_sample(n,m,noise_scal):
    full_corr = []
    full_inc = []
    corr = []
    inc = []
    noise_scal = noise_scal
    for i in range(n):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_corr.append(sig_corr + N)
        corr.append(np.transpose(np.asmatrix(corr_topo)) * np.asmatrix(full_corr[i]))
    for j in range(m):
        N = noise_scal * np.random.normal(size=len(sig_corr))
        full_inc.append(sig_inc + N)
        inc.append(np.transpose(np.asmatrix(inc_topo)) * np.asmatrix(full_inc[j]))
    return corr, inc

correct , incorrect = mk_sample(50,50,0.5)
correct.__len__() #60
correct[0].shape #(275, 1000)

correct_array = np.array(correct) # (60, 275, 1000)
incorrect_array = np.array(incorrect)  # (12, 275, 1000)
'''


# 2. Fit the SVM model
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, ShuffleSplit

# Define an SVM classifier (SVC) with a linear kernel
clf = SVC(kernel='linear')
# Define a monte-carlo cross-validation generator (to reduce variance):
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

# 3. 일단은 time point별 아닌, 그냥 전체 time point of each epoch 가지고 predict.
# The goal is going to be to learn on 80% of the epochs and evaluate on the remaining 20% of trials if we can predict accuratly.
X = np.concatenate((correct_array,incorrect_array)) # (100, 275, 300)
X_2d = X.reshape(len(X), -1) # (100, 82500)
y = [1]*50 + [0]*50

scores_full = cross_val_score(clf, X_2d, y, cv=cv, n_jobs=1)
print("Classification score: %s (std. %s)" % (np.mean(scores_full), np.std(scores_full)))
# Classification score: 0.6399999999999999 (std. 0.15297058540778355)

# 4. Now the separate decoders at each time point
# It's also possible to run the same decoder and each time point to know when in time the conditions can be better classified:
n_times = X.shape[2]
scores = np.empty(n_times) #n_times = the number of time points in an epoch.
std_scores = np.empty(n_times)

for t in range(n_times):
    Xt = X[:,:,t] #At a certain time point, we get all values at epochs and channels
    # Standardize features : Xt 값을 전체 epochs에 대한 평균을 빼고 표준편차로 나눈 값으로 변경.
    Xt -= Xt.mean(axis=0) # Subtract the average over epochs
    Xt /= Xt.std(axis=0) # Divide by the std over epochs
    # Run cross-validation
    scores_t = cross_val_score(clf,Xt,y,cv=cv, n_jobs=1) # at each time point, we run different decoders
    scores[t] = scores_t.mean()
    std_scores[t] = scores_t.std()

# Scaling
times = 1e3 * time_vec # to have times in ms
scores *= 100  # make it percentage accuracy
std_scores *= 100

# Plotting temporal decoding
plt.plot(times, scores, label="Classif. score")
plt.axhline(50, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.axhline(100 * np.mean(scores_full), color='g', label='Accuracy full epoch')
plt.legend()
hyp_limits = (scores - std_scores, scores + std_scores)
#plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
plt.xlabel('Times (ms)')
plt.ylabel('CV classification score (% correct)')
plt.ylim([0, 120])
plt.title('Sensor space decoding')
plt.show()

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)


# Temporal generalization
from mne.decoding import GeneralizingEstimator

# Compute Area Under the Curver (AUC) Receiver Operator Curve (ROC) score
# of time generalization. A perfect decoding would lead to AUCs of 1.
# Chance level is at 0.5.
# The default classifier is a linear SVM (C=1) after feature scaling.

# Define the Temporal generalization object
time_gen = GeneralizingEstimator(clf, n_jobs = 1, scoring='roc_auc', verbose=True)
scores = cross_val_multiscore(time_gen, X, y, cv=3, n_jobs=1) # again, cv=3 just for speed
# Mean scores across cross-validation splits
scores = np.mean(scores, axis=0)

'''# Plot the diagonal (it's exactly the same as the time-by-time decoding above)
fig, ax = plt.subplots()
ax.plot(n_times, np.diag(scores), label='score')
ax.axhline(.5, color='k', linestyle='--', label='chance')
ax.set_xlabel('Times')
ax.set_ylabel('AUC')
ax.legend()
ax.axvline(.0, color='k', linestyle='-')
ax.set_title('Decoding MEG sensors over time')'''

#
times = 1e3 * time_vec # convert times to ms
plt.imshow(scores, interpolation='spline16', origin='lower',
           extent=[time_vec[0], time_vec[-1], time_vec[0], time_vec[-1]],
           vmin=0., vmax=1.)
plt.xlabel('Times Test (ms)')
plt.ylabel('Times Train (ms)')
plt.title('Time generalization (Simulated data)')
plt.axvline(0, color='k')
plt.axhline(0, color='k')
plt.colorbar()