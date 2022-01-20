import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import mne
from mne.datasets import sample
from mne.decoding import (SlidingEstimator, GeneralizingEstimator, Scaler,
                          cross_val_multiscore, LinearModel, get_coef,
                          Vectorizer, CSP)

data_path = sample.data_path()

subjects_dir = data_path + '/subjects'
raw_fname = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
tmin, tmax = -0.200, 0.500
event_id = {'Auditory/Left': 1, 'Visual/Left': 3}  # just use two
raw = mne.io.read_raw_fif(raw_fname)
raw.pick_types(meg='grad', stim=True, eog=True, exclude=())

# The subsequent decoding analyses only capture evoked responses, so we can
# low-pass the MEG data. Usually a value more like 40 Hz would be used,
# but here low-pass at 20 so we can more heavily decimate, and allow
# the example to run faster. The 2 Hz high-pass helps improve CSP.
raw.load_data().filter(2, 20)
events = mne.find_events(raw, 'STI 014')

# Set up bad channels (modify to your needs)
raw.info['bads'] += ['MEG 2443']  # bads + 2 more

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, proj=True,
                    picks=('grad', 'eog'), baseline=(None, 0.), preload=True,
                    reject=dict(grad=4000e-13, eog=150e-6), decim=3,
                    verbose='error')
epochs.pick_types(meg=True, exclude='bads')  # remove stim and EOG
del raw


#----------------------------------------------------------------------------------------------------------
# Check ERF

evoked_Auditory = epochs['Auditory/Left'].average()
evoked_Visual = epochs['Visual/Left'].average()
evoked_contrast = mne.combine_evoked([evoked_Auditory, evoked_Visual], weights='nave')

fig = evoked_Auditory.plot()
fig = evoked_Visual.plot()
fig = evoked_contrast.plot()

# Plot some topographies

times = np.linspace(-0.1, 0.5, 10)
fig = evoked_Auditory.plot_topomap(times=times, ch_type='grad')
fig = evoked_Visual.plot_topomap(times=times, ch_type='grad')
fig = evoked_contrast.plot_topomap(times=times, ch_type='grad')

# -------------------------------------------------------------------------------------------------------------

#### Clssify single trials with an SVM

# 1. Prepare X and y
epochs_list = [epochs[k] for k in event_id]

#len(epochs['Auditory/Left']) Out[44]: 56
# len(epochs['Visual/Left']) Out[45]: 67
mne.epochs.equalize_epoch_counts(epochs_list) #To have a chance at 50% accuracy equalize epoch count in each condition

# A classifier takes as input an x and return y (-1 or 1).
# Here x will be the data at one time point on all gradiometers (hence the term multivariate).
# We work with all sensors jointly and try to find a discriminative pattern between 2 conditions to predict the class.

n_times = len(epochs.times)

# Take only the data channels (here the gradiometers)
data_picks = mne.pick_types(epochs.info, meg=True, exclude='bads')

# Make arrays X and y such that :
# X is 3d (n_epochs, n_meg_channels, n_times) with X.shape[0] is the total number of epochs to classify
# y is filled with integers coding for the class to predict
# We must have X.shape[0] equal to y.shape[0]

X = [e.get_data()[:, data_picks, :] for e in epochs_list]
y = [k * np.ones(len(this_X)) for k, this_X in enumerate(X)] # To check which step we are at
X = np.concatenate(X)
y = np.concatenate(y)


# 이게 아주 간단하게 아래와 같이 바뀜
x1 = epochs_list[0].get_data()
x2 = epochs_list[1].get_data()
X = np.concatenate((x1,x2))

y = np.append(np.zeros(len(x1)),np.ones(len(x2)))
'''X.shape
Out[102]: (112, 203, 36)'''


''' 참고) 이 epoch는 1,3 말고 다른 epoch도 포함,, 내생각. the number of epochs 수가 더 많아
X = epochs.get_data()  # MEG signals: n_epochs, n_meg_channels, n_times
y = epochs.events[:, 2]  # target: auditory left vs visual left
X.shape = (123(the number of epochs), 203(the number of channels), 36(the number of time points in an epoch))
y.shape = (123,)'''

# 2. Fit the SVM model
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, ShuffleSplit

# Define an SVM classifier (SVC) with a linear kernel
clf = SVC(kernel='linear')
# Define a monte-carlo cross-validation generator (to reduce variance):
cv = ShuffleSplit(10, test_size=0.2, random_state=42)

# 3. 일단은 time point별 아닌, 그냥 전체 time point at each epoch 가지고 predict.
# The goal is going to be to learn on 80% of the epochs and evaluate on the remaining 20% of trials if we can predict accuratly.
X_2d = X.reshape(len(X), -1) # 행이 112개, 열은 알아서 자동으로 재배열해주는 함수
X_2d = X_2d / np.std(X_2d) # Scaling?

scores_full = cross_val_score(clf, X_2d, y, cv=cv, n_jobs=1)
print("Classification score: %s (std. %s)" % (np.mean(scores_full), np.std(scores_full)))
# Classification score: 0.9956521739130434 (std. 0.013043478260869556)

# 4. Now the separate decoders at each time point
# It's also possible to run the same decoder and each time point to know when in time the conditions can be better classified:

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
times = 1e3 * epochs.times # to have times in ms
scores *= 100  # make it percentage accuracy
std_scores *= 100

# Plotting temporal decoding
plt.plot(times, scores, label="Classif. score")
plt.axhline(50, color='k', linestyle='--', label="Chance level")
plt.axvline(0, color='r', label='stim onset')
plt.axhline(100 * np.mean(scores_full), color='g', label='Accuracy full epoch')
plt.legend()
hyp_limits = (scores - std_scores, scores + std_scores)
plt.fill_between(times, hyp_limits[0], y2=hyp_limits[1], color='b', alpha=0.5)
plt.xlabel('Times (ms)')
plt.ylabel('CV classification score (% correct)')
plt.ylim([30, 100])
plt.title('Sensor space decoding')

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
times = 1e3 * epochs.times # convert times to ms
plt.imshow(scores, interpolation='spline16', origin='lower',
           extent=[times[0], times[-1], times[0], times[-1]],
           vmin=0., vmax=1.)
plt.xlabel('Times Test (ms)')
plt.ylabel('Times Train (ms)')
plt.title('Time generalization (%s vs. %s)' % tuple(event_id.keys()))
plt.axvline(0, color='k')
plt.axhline(0, color='k')
plt.colorbar()