import autograd.numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from autograd import grad

# get a function to find the minimum of
x, y = np.linspace(0, 10, 100), np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)

#gaussian distribution
normal = lambda x, mu, s, c: c*(1/np.sqrt(s*2*np.pi))*np.exp(-0.5*(x-mu)**2)

# 1d normal
t = lambda x: normal(x, 5, 1, -1.)

# convoluted 3d function
cv = lambda X, Y: normal(X, 5, 1, -1.)*normal(Y, 7, 0.2, 2) + normal(X, 3, 1, -2)*normal(Y ,3, 1, -2) + normal(X, 8, 1, -1)*normal(Y, 6, 0.2, 1)


def descend(f, p0, learn=0.1, max_i=100):
    """
    function to plot the progess of 1d grad descent on function
    returns the minimum found and number of iterations needed
    """
    plt.plot(p0, f(p0), 'ro')
    i = 0
    p1 =  None
    grad_f = grad(f) #use autograd library to find the gradient of the function
    while i < max_i and p1 !=p0:
        p1 = p0
        p0 = p0 - learn*grad_f(p0)
        plt.plot(p0, f(p0), 'ko', alpha=0.1)
        i += 1
    return p0, i



def descend_3d(f, p0, learn=0.01, max_i=100):
    """
    same as previous function but in 3d
    """
    ax = plt.gca()
    ax.scatter(p0[0], p0[1], zs=f(p0[0], p0[1]), c='r', s=20)
    plt.savefig('descent_3d/000.png')
    i = 0
    p1 =  None
    grad_f = grad(f) #use numpy to get gradient of data
    while i < max_i:
        p1 = p0
        g = grad_f(p0[0], p0[1])
        p0[0] = p0[0] - learn*g
        p0[1] = p0[1] - learn*g
        print(p0)
        ax.scatter(p0[0], p0[1], zs=f(p0[0], p0[1]), c='k', alpha=0.7, s=10)
        i += 1
        plt.savefig('descent_3d/{:03d}.png'.format(i))
    return p0, i

def descend_3d_mom(f, p0, learn=0.01, gamma=0.05, max_i=100):
    ax = plt.gca()
    ax.scatter(p0[0], p0[1], zs=f(p0[0], p0[1]), c='r', s=50)
    plt.savefig('descent_3d/000.png')
    i = 0
    v = 0
    p1 =  None
    grad_f = grad(f) #use numpy to get gradient of data
    while i < max_i:
        p1 = p0
        g = grad_f(p0[0], p0[1])
        v = gamma*v + learn*g
        p0[0] = p0[0] - v
        p0[1] = p0[1] - v
        print(p0)
        ax.scatter(p0[0], p0[1], zs=f(p0[0], p0[1]), c='k', alpha=0.7, s=10)
        i += 1
        plt.savefig('descent_3d/{:03d}.png'.format(i))
    return p0, i

def plt_plot_3dnormal(x, y, z):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, y, z, cmap=cm.YlGnBu, linewidth=0, alpha=0.3, antialiased=True)

    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_zlabel('z')
    plt.show()
