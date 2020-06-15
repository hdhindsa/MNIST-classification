import numpy as np
import struct
import matplotlib.pyplot as plt
from scipy.special import expit

#####Manually change the size of training dataset in readMNISTdata() to reproduce results

#### This is the L2 regularized code

# MNIST dataset is available at http://yann.lecun.com/exdb/mnist/
# after downloading, extract all four files
def readMNISTdata():

    with open('t10k-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))
    
    with open('t10k-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size,1))
    
    with open('train-images.idx3-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))
    
    with open('train-labels.idx1-ubyte','rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size,1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate( ( np.ones([train_data.shape[0],1]), train_data ), axis=1)
    test_data  = np.concatenate( ( np.ones([test_data.shape[0],1]),  test_data ), axis=1)
    np.random.seed(314)
    np.random.shuffle(train_labels)
    np.random.seed(314)
    np.random.shuffle(train_data)

    X_train = train_data[:300] / 256
    t_train = train_labels[:300]

    X_val   = train_data[50000:] /256
    t_val   = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data, test_labels




def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    #z = transpose(w)*x
    #y = softmaz(z)
    #t_hat = argmax y

    # use y for loss which is probabilities of each label
    #use that for accuracy
    

    #calculate y
    z = np.dot(X,W)
    max_z = np.transpose(np.amax(z,axis=1))
    
    max_z = np.tile(max_z,(z.shape[1],1))
    max_z = np.transpose(max_z)
    z = np.subtract(z,max_z)

    z = np.exp(z)
    y_deno = (np.sum(z, axis = 1))
    y_deno = np.tile(y_deno,(z.shape[1],1))
    y_deno = np.transpose(y_deno)
   
    y = np.divide(z,y_deno)

    
    #t_hat calculate
    t_hat = np.zeros_like(y)
    t_hat[np.arange(len(y)), y.argmax(1)] = 1

    #loss 
    t_indices = np.amax(t,axis=1)
    
    log_y = np.log(y)
    array_of_log_y = -log_y[np.arange(len(y)), t_indices]
    loss = (array_of_log_y.sum())/y.shape[0]
    

    #accuracy
    t_indices = np.amax(t,axis=1)
    new_t = np.zeros_like(y)
    new_t[np.arange(len(y)), t_indices] = 1
    
    part1 = t_hat[new_t == 1]
    
    part2 = part1[part1 == 1]
    correct = part2.sum()
    acc = correct/t.shape[0]
    
    return y, t_hat, loss, acc



def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    
    W = np.zeros([X_train.shape[1], 10])
    # w: (d+1)x1

    losses_train = []
    acc_val   = []

    W_best    = None
    acc_best = 0
    epoch_best = 0
    
    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range( int(np.ceil(N_train/batch_size)) ):
            
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            y_hat_batch, _, loss_batch, _ = predict(X_batch, W, y_batch)
            loss_this_epoch += loss_batch

            
            # Mini-batch gradient descent

            t_indices = np.amax(y_batch,axis=1)
            new_t = np.zeros_like(y_hat_batch)
            new_t[np.arange(len(y_hat_batch)), t_indices] = 1

            
            diff = np.subtract(y_hat_batch,new_t)
            grad = np.dot(np.transpose(X_batch), diff)
            grad = grad/batch_size
            W_old = W
            decay_W = np.multiply(decay,W)
            grad_updated = np.add(grad,decay_W)
            W = np.subtract(W, np.multiply(alpha,grad_updated))

        
        # monitor model behavior after each epoch
        # Compute the training loss by averaging loss_this_epoch
        losses_train.append(loss_this_epoch/batch_size)
        # Perform validation on the validation test by the risk
        _, _,_, val_acc = predict(X_val, W_old, t_val)
        acc_val.append(val_acc)
        # Keep track of the best validation epoch, risk, and the weights
        
        if val_acc > acc_best:
            acc_best = val_acc
            epoch_best = epoch
            W_best = W_old
    
    return epoch_best, acc_best,  W_best, losses_train, acc_val


##############################

X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)



N_class = 10

alpha   = 0.1      # learning rate
batch_size   = 100    # batch size
MaxIter = 50        # Maximum iteration
decay = 0.2          # weight decay


epoch_best, acc_best, W_best,losses_train, acc_val  = train(X_train, t_train, X_val, t_val)

_, _, _, acc_test = predict(X_test, W_best, t_test)


print('At epoch', epoch_best, 'val: ', acc_best, 'test:', acc_test)#, 'train:', acc_train)




plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
risk_epoch = range(1,51)
plt.plot(risk_epoch, acc_val , color="blue")

#plt.legend()
plt.tight_layout()
plt.savefig('Risk_q2.png')


plt.figure()
risk_epoch = range(1,51)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.plot(risk_epoch, losses_train, color="blue")
#plt.legend()
plt.tight_layout()
plt.savefig('losses_q2.png')
