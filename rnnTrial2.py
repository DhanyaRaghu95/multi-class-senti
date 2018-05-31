"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import pickle

# data I/O
#data = open('input1.txt', 'r').read() # should be simple plain text file
#chars = list(set(data))
#data_size, vocab_size = len(data), len(chars)

#char_to_ix = { ch:i for i,ch in enumerate(chars) }
#ix_to_char = { i:ch for i,ch in enumerate(chars) }

count=0

dataall = pickle.load(open("phrases_train.p",'r'))#open('input1.txt', 'r').read() # should be simple plain text file
targetsall = pickle.load(open("sentiments_train_five.p",'r'))
dataall_test = pickle.load(open("phrases_test.p",'r'))#open('input1.txt', 'r').read() # should be simple plain text file
targetsall_test = pickle.load(open("sentiments_test_five.p",'r'))
#targetsall = pickle.load(open("senti3.p",'r'))
print type(targetsall), len(targetsall), len(dataall), targetsall[1], len(dataall)
#exit(0) '''
#print 'data has %d characters, %d unique.' % (data_size, vocab_size)
# hyperparameters
hidden_size = 8 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 0.001
vocab_size = 100
# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(5, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((5, 1)) # output bias

print "random", Wxh[0]
def lossFun(inputs, targets, hprev):
  global count
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    #print "xs" , xs[t]
    xs[t] = np.copy([[i] for i in inputs[t]])
    #print "xs input " , xs[t], np.shape(xs[t])
    #exit(0)
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    #print len(hs[t])
  ys = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
  #print "ys[t]",ys[t]
  ps = ys #np.exp(ys) / np.sum(np.exp(ys)) # probabilities for next chars
  #print "ps",np.argmax(max(ps))
  #print np.shape(ps), ps, ps[0]
  #exit(0)
  #print "ps[t]",ps,[targets], ps[targets,0]
  loss += -ps[targets,0]#-np.log(ps[targets,0]) # softmax (cross-entropy loss)
  #print loss
  #exit(0)
  #print "------",ps[t][targets[t],0]
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])

  dy = np.copy(ps)  #print "dy",dy
  #exit(0)
  dy[targets] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
  #print ps,dy,  targets
  maxp = max(ps)
  i=0
  for p in ps:
    if p==maxp:
        #print "predicted : ",i,"   target : ",targets
        if i == targets:
            count+=1
            #print "predicted : ",i,"   target : ",targets
    i+=1
  #exit(0)
  #print np.shape(dy), np.shape(hs[t].T)

  dWhy += np.dot(dy, hs[t].T)
  dby += dy
  #exit(0)
  for t in reversed(xrange(len(inputs))):
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  #exit(0)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  #print loss
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]
'''
def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes
'''
n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
while n<144220:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(dataall) or n == 0:
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
  #inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  #targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
  inputs = dataall[n]
  #print "i ", inputs
  targets = [float(targetsall[n])]
  targets = targetsall[n]
  #print "t ",targets
  # sample from the model now and then
  '''if n % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )
  '''
  # forward seq_length characters through the net and fetch gradient
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001
  if n % 100 == 0: print 'iter %d, loss: %f' % (n, smooth_loss) # print progress

  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update

  p += seq_length # move data pointer
  n += 1 # iteration counter

print count
acc=0
c0=0
c1=0
c2=0
c3=0
c4=0
def predict(inputs, target):
  global count,acc,c0,c1,c2,c3,c4
  hs={}
  #xs, hs, ys, ps = {}, {}, {}, {}
  #print "learnt", Wxh[0]
  #exit(0)
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
   # xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    #print "xs" , xs[t]
    x = np.copy([[i] for i in inputs[t]])
    #print "xs input " , xs[t], np.shape(xs[t])
    #exit(0)
    hs[t] = np.tanh(np.dot(Wxh, x) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    #print len(hs[t])
  ys = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
  #print "ps",np.argmax(ys),target,type(np.argmax(ys)), type(target)
  ps = ys
  if(np.argmax(ps))==0:
      c0+=1
  if(np.argmax(ps))==1:
      c1+=1
  if (np.argmax(ps)) ==2:
      c2 += 1
  if (np.argmax(ps)) ==3:
      c3 += 1
  if (np.argmax(ps)) ==4:
      c4 += 1
  if (np.argmax(ps) == int(target)):
    acc += 1
    #print acc,"#######"


for n in range(54000):
  inputs = dataall_test[n]
  target = targetsall_test[n]
  predict(inputs, target)
print "Accuracy", acc, float(acc) / 54000 * 100
print "c",c0,c1,c2,c3,c4