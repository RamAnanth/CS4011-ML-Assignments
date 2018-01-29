import numpy as np
import matplotlib.pyplot as plt

x=[i for i in range(1,4)]
y = [0.184107456577,0.210608763551,0.0291593956837]
algo_names = ['GaussianNB','MultinomialNB','BernoulliNB']

plt.plot(x,y,'g-')
plt.title('Performance of different Naive Bayes Classifiers')
plt.xlabel('Classifier')
plt.ylabel('F-Score')
plt.xticks(algo_names)
plt.show()

Multi:5
Bernoulli:5
