ptf %
import numpy as np
import cPickle as pickle

# define objective function: returns the array to be minimized
def sine_decay():
    """ model decaying sine wave, subtract data"""
    amp = %amp%
    shift = %shift%
    omega = %omega%
    decay = %decay%

    x = np.linspace(0, 15, 301)
    model = amp * np.sin(x * omega + shift) * np.exp(-x*x*decay)

    # Waste time on purpose
    for i in range(10**6): x = np.sin(i)

    obsnames = ['obs'+str(i) for i in range(1,model.shape[0]+1)]
    return dict(zip(obsnames,model))


if __name__== "__main__":
    out = sine_decay()
    pickle.dump(out,open('sine.pkl', 'wb'))


