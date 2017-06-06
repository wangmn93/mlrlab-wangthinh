import numpy as np
import math
from numpy.linalg import inv
import gym
import random
from matplotlib import  pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate.
    np.meshgrid is pretty annoying!
    """
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])


def valuePlot(weight):
    X_grid = grid2d(-1, 1, num=30)
    y_grid = []
    for x in X_grid:
        action = getAction(range(8),weight,x,getFeature)
        y_grid.append(np.dot(weight,getFeature(x,action)))
    y_grid = np.array(y_grid)
    # vis the result
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_grid[:, 0], X_grid[:, 1], y_grid)

    ax.set_title("Value function")
    plt.show()

def mdot(*args):
    """Multi argument dot function. http://wiki.scipy.org/Cookbook/MultiDot"""
    return reduce(np.dot, args)

#input type np.array
def RBF(center, covariance, x):
    d_2 = mdot((x-center).T,inv(np.diag(covariance)),x-center)

    return math.exp(-d_2/2.)


#input np.array
def getFeature(x,action,n =8, m=8, range1 =(-1,1),range2=(-1,1),covariance = np.array([.04, .04])):
    p_centers,v_centers = generateCenter(range1,n,range2,m)
    feature = [1]#append one
    for i in p_centers:
        for j in v_centers:
            rbf = RBF(np.array([i,j]),covariance,x)
            feature.append(rbf)
    finalFeature = []
    for i in range(8):
        if i == action:
            finalFeature+=feature
        else:
            finalFeature+=[0]*65
    return np.array(finalFeature)



def generateCenter(range1,n, range2, m):
    range1_span = (range1[1]-range1[0])/float(n-1)
    range2_span = (range2[1]-range2[0])/float(m-1)
    center1 = [range1[0]+i*range1_span for i in range(n)]
    center2 = [range2[0]+i*range2_span for i in range(m)]
    return center1,center2


def sampleFromStateSpace():
    distance2D = random.uniform(0,1)
    angle = random.uniform(-math.pi,math.pi)
    return np.array([angle,distance2D])

def initialSample():
    x = random.uniform(-1,1)
    y = random.uniform(-1,1)
    return np.array([x,y])

def getReward(nState):
    difference = math.sqrt(np.dot(nState,nState))
    if difference < .05:
        return 1
    else:
        return -1

def getAction(actions,weight,state,feature_function):
    values = [np.dot(weight,feature_function(state,action))for action in actions]
    return values.index(max(values))

def transition(state,action,actions):
    nState = state-actions[action]
    nState[0] = max(min(1,nState[0]),-1)
    nState[1] = max(min(1,nState[1]),-1)
    reward = getReward(nState)
    return nState,reward

def sampleFromOneEpisode(k):
    exp_list = []
    cState = initialSample()
    success = 0
    for i in range(k):
        action = random.randint(0,8)
        nState,reward = transition(cState,action,actions)
        exp_list.append((cState,action,reward,nState))
        #plt.scatter(nState[0], nState[1])
        cState = nState
        if reward ==1:
            success +=1
    return exp_list,success

if __name__ == "__main__":
    actions = []
    stepSize = .01
    for i in [stepSize,-stepSize,0]:
        for j in [stepSize,-stepSize,0]:
            action = np.array([i,j])
            actions.append(action)

    exp_list,success = sampleFromOneEpisode(900)
    #print getFeature(cState,0).shape
    #plt.ion()
    print "Stop sample, success",success
    print "Sample steps",len(exp_list)
    ########################LSPI########################
    old_weight = np.zeros(520)
    difference = 10
    while difference>1:
        A_tilde = np.eye(520)*0.1 #initialize A
        b_tilde = np.zeros(520) #initialize b
        for e in exp_list:
            nAction = getAction(range(8), old_weight, e[3], getFeature)#select action for next state according to current weight
            feature = getFeature(e[0],e[1])
            reward = e[2]
            nFeature = getFeature(e[3],nAction)
            diff = feature-.99*nFeature
            A_tilde += np.outer(feature,diff.T)
            b_tilde += feature*reward
        weight = np.dot(inv(A_tilde),b_tilde)
        difference = np.dot(old_weight - weight, old_weight - weight)#compute the diff between two weights
        old_weight = np.copy(weight)
        print "delta",difference
    #plot

    #test
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Sample data&Test")
    for exp in exp_list:
        nState = exp[3]
        ax.scatter(nState[0], nState[1], color='b')

    cState = initialSample()
    print "ini ", cState
        # print getFeature(cState,0).shape
        # plt.ion()
    steps = 0
    for i in range(100):
        action = getAction(range(8),old_weight,cState,getFeature)
        nState, reward = transition(cState, action, actions)
        print nState
        exp_list.append((cState, action, reward, nState))
        ax.scatter(nState[0], nState[1],color='r')
        cState = nState
        steps+=1
        if reward == 1:
            print "Success"
            break
    print "Stop test"
    print steps
    plt.show(block=False)

    valuePlot(old_weight)







    # for i in range(5):
    #     a = random.randint(0,8)
    #     print actions[a]
    #     transition(cState,a,actions)


