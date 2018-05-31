import pickle
listOfSent=[]
with open('phrase_test_classes.txt', 'rt') as csvfile:
    reader = csvfile.readlines()
for i in reader:
    listOfSent.append(i.strip().split("|")[1])
print listOfSent
print len(listOfSent)
pickle.dump(listOfSent,open("sentiments_test_five.p","w"))