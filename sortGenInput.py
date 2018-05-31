import random

dataall=[]
targetall=[]
for j in range(1000):
    dataall.append([])
    for i in range(15):
        dataall[j].append(random.randint(1,500))
    if(j%3==0):
        dataall[j].sort()
        targetall.append(1)
    else:
        targetall.append(0)
# 0 if not sorted, 1 if sorted
for j in range(1000):
    random.shuffle(zip(dataall[j],targetall))

for i in range(10):
    print dataall[i],targetall[i]