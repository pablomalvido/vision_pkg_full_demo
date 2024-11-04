import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.use('agg')

def plot_cables(cables, name, img_shape):
    fig, ax = plt.subplots()
    for cable in cables:
        inverted_line = [(i*-1)+img_shape[0] for i in cable['y']]
        color_corrected = (float(cable['color'][2])/255, float(cable['color'][1])/255, float(cable['color'][0])/255)
        ax.plot(cable['x'], inverted_line, color=color_corrected)
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.ylim([-y_limits[1],-y_limits[0]])
    ax.set_aspect('equal')
    #ax.set_adjustable("datalim")
    plt.ylim([0,img_shape[0]])#200,1000
    #plt.xlim([-30,780])#-30,1040
    plt.title('Cables shape estimation')
    plt.grid(True)
    ax.set_facecolor('violet')
    plt.savefig(os.path.dirname(os.path.realpath(__file__)) + "/../plots/" + str(name.split('.')[0] + ".pdf"))
    plt.rcdefaults()