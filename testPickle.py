import pickle
model = pickle.load(open("sentiments_train_five.p","r"))
#model2 = pickle.load(open("sentiments_test.p","r"))
#pickle.dump(model[:10000],open('sentiments_test.p','w'))
#print model[0],len(model),len(model[0][0])
#print model2[0],len(model2),len(model2[0])
print model