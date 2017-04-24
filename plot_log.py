import pickle

import matplotlib.patches as mpatches
import matplotlib.pylab as plt

log_fa = pickle.load(open("log_fa.p", "rb"))
log_q = pickle.load(open("log_q.p", "rb"))

x = xrange(1, len(log_fa))
y_fa = [e[0] for e in log_fa[1:]]
y_q = [e[1][-1] for e in log_q]

plt.plot(x, y_fa, 'r', x, y_q, 'b')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Learning Curves')

plt.legend(handles=[mpatches.Patch(color='red', label='LFA'),
                    mpatches.Patch(color='blue', label='Q-learning')])

plt.figure()
plt.bar(xrange(len(log_fa[-1][1])), log_fa[-1][1])
plt.xlabel('Feature')
plt.ylabel('Weight Value')
plt.title('Feature Weights')

plt.figure()
for f in xrange(len(log_fa[-1][1])):
    traces = [e[1][f] for e in log_fa[1:]]
    plt.plot(traces)
plt.xlabel('Episode')
plt.ylabel('Weight Value')
plt.title('Weights Traces')

plt.show()

