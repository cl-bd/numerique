import numpy as np;
import sympy
import matplotlib.pyplot as plt;
import matplotlib.animation as animation;
from mpl_toolkits.mplot3d import Axes3D

# def des variables et fonctions tension / intensité
x,t,L,C,R,G,c = sympy.symbols('x t L C R G c',positive=True) # create symbols, these won't be negative along the domain
v = sympy.Function('v')(x,t) # fonction tension
i = sympy.Function('i')(x,t) # fonction tension
c = L*C # velocite

def eq_onde(f,L,C,R,G):
    f_t = sympy.diff(f,t,1) # 1re derivee tension
    f_tt = sympy.diff(f,t,2) # 2eme derivee tension
    f_xx = sympy.diff(f,x,2) # get second space derivative
    return sympy.Eq(f_xx-L*C*f_tt-(R*C+L*G)*f_t) # equation d'onde


eq_v = eq_onde(v,L,C,R,G)
print("\nEquation sur la tension : ")
sympy.pprint(eq_v)
print("\nEquation sur l'intensité : ")
eq_i = eq_onde(i,L,C,R,G)
sympy.pprint(eq_i)



def islambda(f):
    LAMBDA = lambda:0
    return isinstance(f, type(LAMBDA)) and f.__name__ == LAMBDA.__name__

# Definition of ODE solver by FDM
def fdm(A = 0, B = 0, C = 0, D = 0, y0 = 0, dydx = 1, h = 0.5, max_x = 5, drawplot = True):
    iterations = int(max_x / h);
    x = [];
    y = [];
    
    y.append(0);
    y.append(y[0] + dydx * h)
    
    x.append(0);
    x.append(h);
    
    for i in range(2, iterations):
        x.append(x[i - 1] + h)
        X = x[i]
        
        a = A(X) if islambda(A) else A
        b = B(X) if islambda(B) else B
        c = C(X) if islambda(C) else C
        d = D(X) if islambda(D) else D
            
        y.append((d - a * ((-2 * y[i - 1] + y[i - 2]) / h**2) + b * (y[i - 2] / (2 * h)) - c * y[i - 1]) * (2 * h**2 / (2 * a + h * b)))
    
    if(drawplot):
        plt.plot(x, y)
        plt.show()
    
    print("Number of iterations:", iterations)
    print("Last x value:", x[len(x) - 1])
    print("Last y value:", y[len(y) - 1])
    

# Test call of ODE solver with appropriate settings
#fdm(1, -5, 4, lambda x: x**2, 0, 1, 0.005, 5, True)

#fdm(1, 0, 3, 0, 0, 1, 0.005, 5, True)

def I(x):
    return 0;

def gen(t):
    f=50    # frequence en Hz
    A=225    # amplitude source en V
    return A*np.sin(2*np.pi*f*t);

def h(t):
    return 20*np.sin(t);

def wave():
    import numpy as np;
    L=1.3e-6   #(source RTE) : 1,3 mH/km
    C=10e-12   #(source RTE) : 10 nF/km
    R=0.04e-3*1000  #(source RTE) : 0,04 Ω/km
    G=0             #(source RTE) : 10 nF/km
    c = 1/np.sqrt(L*C);
    l = 5000e3; #longueur ligne
    dx = 20e3;   #pas spatial
    dt = 0.00002;#pas temps
    K = c*dt/dx;
    
    max_y = 500;
    max_t = 3000*dt;
    
    t_res = int(max_t / dt);
    x_res = int(l / dx);
    
    frameinterval= 1;
    
    x_space = np.linspace(0, l, x_res);
    x_space_km = 0.001*x_space
    t_space = np.linspace(0, max_t, t_res);
    
    u = [[0 for t in range(len(x_space))] for x in range(len(t_space))]
    
    #Set initial conditions
    for i in range(0, len(x_space)):
        u[0][i] = I(x_space[i]);
    
    #Incorporate du/dt = 0
    for i in range(1, len(x_space) - 1):
        u[1][i] = u[0][i] - 1/2*K**2*(u[0][i+1] - 2*u[0][i] + u[0][i-1])
    
    # Enumeration trough definition-space to get both value and index
    for n, t in enumerate(t_space):
        # compute x for this time
        for i, x in enumerate(x_space):
            # boundary conditions
            u[n][0] = gen(t);                # can be a function of t
            u[n][len(x_space)-1] = 0;   # can be a function of t
            
            
            if n > 1 and n < len(t_space) - 1 and i < len(x_space) - 1:
                pertes = R*G*dt**2*u[n][i]-(R*C+L*G)/2*dt*u[n-1][i]
                u[n+1][i] = ( K**2*(u[n][i+1] - 2*u[n][i] + u[n][i-1]) + 2*u[n][i] - u[n-1][i] + pertes ) / (1+(R*C+L*G)*dt/2)
           
            
    
    def init():
        line.set_ydata(u[0])
        return line
        
    def animate(i):
        line.set_ydata(u[2*i])
        title = "t : " + str(round(t_space[i]*1000,2)) + " ms  --  i : " + str(i)
        ax.set_title(title)
        return line
    
    # Animated output
    fig, ax = plt.subplots()
    ax.set_xlabel('Distance km')
    ax.set_ylabel('Tension kV')
    line, = ax.plot(x_space_km, u[0])
    plt.ylim(-max_y, max_y)
    ani = animation.FuncAnimation(fig, animate, init_func=init, interval=frameinterval, blit=False, save_count=len(t_space))
    plt.plot(ani)
    #ani.save("wave_smooth.avi")
    #ani.save() # This throws an error because no filename, but the animation will not play without this
    
    #2D static output
#    for i in range(len(u)):
#        plt.plot(u[i])
#        
#    plt.show()
    
#    #3D static output
#    x = range(len(x_space));
#    y = range(len(t_space));
#    X, Y = np.meshgrid(x, y);
#    fig = plt.figure();
#    ax = fig.gca(projection='3d')
#    data = np.array(u);
#    ax.plot_surface(X, Y, data);


wave()
