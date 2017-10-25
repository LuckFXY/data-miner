import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

def generator_line_data():
    np.random.seed(8)
    array  = np.random.randn(20,2)
    X = np.r_[array - [3,3], array+[3,3]]
    y = np.array([-1]*20 + [1]*20).reshape(-1,1)
    return X,y

def show_line_svm_2d(X,y,w,b,support_vectors=None):
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x2_max),np.linspace(x2_min, x2_max))
    #w = clf.coef_[0]
    print(xx1.shape)
    #f = w[0]*X[:, 0] + w[1]*X[:, 1] + clf.intercept_[0] + 1
    f = w[0]*xx1 + w[1]*xx2 + b + 1
    plt.contour(xx1,xx2,f,[0,1,2], colors='r')
    #plt.contour(xx1,xx2,f,[0,], colors='r')
    plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.Paired)
    if type(support_vectors) != type(None):
        plt.scatter(support_vectors[:,0],support_vectors[:,1],color='k')
    plt.show()

def selectJrand(i,m):
    j=i
    while(j==i):
        j = int(np.random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if(aj>H):
        return H
    if(aj<L):
        return L
    return aj
from numpy import shape, dot
def smoSimple(dataMat, label, C, toler, maxIter):
    '''
        dataMat   : matrix , the features matrix
        labels : array  , the labels vector of class
        C           :
        toler       : float  , the toler of loss
        maxInter    : int    , the max iter num
    '''
    label = label.astype(np.float32)
    b=0; m,n = shape(dataMat)
    alphas = np.zeros((m,1),dtype=np.float32)
    iter = 0
    while(iter < maxIter):
        alphaPairsChanged = 0
        for i in range(m):
            fXi = dot( dot((alphas * label).T , dataMat) , dataMat[i,:].T ) + b
            Ei = fXi - label[i]
            if ((label[i]*Ei > toler and alphas[i] > 0) or (label[i]*Ei < -toler and  alphas[i]<C) ):
                j = selectJrand(i,m) #choose another alpha
                fXj = dot( dot((alphas * label).T , dataMat) , dataMat[j,:].T ) + b
                Ej = fXj -label[j]
                alphaI_old = alphas[i].copy()
                alphaJ_old = alphas[j].copy()
                if (label[i] != label[j]):          #Make sure the Alpha is less than C
                    L = max(0, alphas[j] - alphas[j]) #
                    H = min(C, C + alphas[j] - alphas[i]) #
                else:
                    L = max(0, alphas[j] + alphas[i] -C)
                    H = min(C, alphas[j] + alphas[i])
                if  L==H :
                    print("L==H")
                    continue
                x1 = dataMat[i,:]
                x2 = dataMat[j,:]
                eta = 2.0 * dot(x1,x2.T) - dot(x1,x1.T) - dot(x2,x2.T)
                if eta >=0:
                    print('eta>=0')
                    continue
                alphas[j] -= label[j] * (Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j], H, L)
                if(abs(alphas[j] - alphaJ_old) < 1e-5):
                    print('J not moving enough')
                    continue
                alphas[i] += label[j] * label[i] * (alphaJ_old - alphas[j]) # update i with the same value of j in negative direction
                b1 = b - Ei - label[i] * (alphas[i] - alphaI_old) * dot(x1,x1.T) - \
                              label[j] * (alphas[j] - alphaJ_old) * dot(x1,x2.T)
                b2 = b - Ej - label[i] * (alphas[i] - alphaI_old) * dot(x1,x2.T) - \
                              label[j] * (alphas[j] - alphaJ_old) * dot(x2,x2.T)
                if(0 < alphas[i] and alphas[i] < C):
                    b=b1
                elif(0 < alphas[j] and alphas[j] < C):
                    b=b2
                else:
                    b = (b1 + b2) / 2.
                alphaPairsChanged += 1
                print("iter %d i: %d, pairs chagend %d"%(iter,i,alphaPairsChanged))
        #------end for
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter =0
        print("iteration number : %d"% iter)
    return b, alphas

if __name__ == '__main__':
    X, y = generator_line_data()
    b, alphas = smoSimple(X, y, 1, 1e-2, 200)
    w = dot(X.T , alphas * y )
    print("w =\n%s \nb = %f"%(str(w),b))
    index = np.argwhere(alphas>0)[:,0]
    print("alpha = ",alphas[index])
    print(index)
    support_vectors =  X[index]
    print(support_vectors)
    show_line_svm_2d(X, y, w, b, support_vectors)

