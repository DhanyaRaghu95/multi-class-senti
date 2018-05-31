import pickle
import numpy as np
import pylab as pl
model = pickle.load(open("sentiments_large.p","r"))
sentiThree=[]
for i in model:
    #print float(i)
    if float(i)>=0 and float(i) <0.2:
        sentiThree.append(0)
    elif float(i) >= 0.2 and float(i) < 0.4:
        sentiThree.append(1)
    elif float(i) >= 0.4 and float(i) <0.6:
        sentiThree.append(2)
    elif float(i) >= 0.6 and float(i) < 0.8:
        sentiThree.append(3)
    elif float(i) >= 0.8 and float(i) <=1:
        sentiThree.append(4)

print sentiThree
#print len(sentiThree)
pickle.dump(sentiThree,open("sentiments_train_five.p","w"))
print len(model)