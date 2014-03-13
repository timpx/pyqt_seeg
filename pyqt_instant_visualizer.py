from pylab import *
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from matplotlib import cm
from scipy.signal import butter, lfilter
import os

print('load_data')
path = '/disk2/WorK/Simulations/TVB/'
## Surface
verts = np.load(os.path.joint(path, 'pyqt_data/verts.npy')
faces = np.load(os.path.joint(path, 'pyqt_data/faces.npy')
centres = np.loadtxt(os.path.joint(path, 'pyqt_data/centres.txt')
vertex_mapping = np.load(os.path.joint(path, 'pyqt_data/vertex_mapping.npy')  
g = open(os.path.joint(path, 'pyqt_data/name_regions.txt','r')
global name_regions
name_regions = []
for line in g:
    name_regions.append(line)
g.close()

## TAVG
x01_curr = 2.5
x02_curr = 3.0
num_curr = 4.0
Ks_curr = 0.5
x0_curr = 69
global tavgs
tavgs = np.load(os.path.join(path, 'sEEG_sim/results/compressed_different_'+str(x02_curr)+
                '_epileptogenic_'+str(x01_curr)+'_net_'
                +str(num_curr)+'_Ks_'+str(Ks_curr)+'_x0_'+str(x0_curr)+'.npy'))

## projection matrix and seegs
proj_mat = np.load('seeg_projection_matrix.npy')
seegs_not_filtered = np.dot(proj_mat, tavgs)
def butterworth_bandpass(lowcut, highcut, fs, order=5):
    """
    Build a diggital Butterworth filter

    """
    nyq  = 0.5 * fs        # nyquist sampling rate
    low  = lowcut / nyq    # normalize frequency
    high = highcut / nyq   # normalize frequency
    b, a = butter(order, [low, high], btype='band')
    return b, a


def filter_data(data, lowcut, highcut, fs, order=5):
    # get filter coefficients
    b, a = butterworth_bandpass(lowcut, highcut, fs, order=order)
    # filter data
    y = lfilter(b, a, data)
    return y

fs = 110.0
lowcut = 0.1   
highcut = 50.0
seegs = filter_data(seegs_not_filtered, lowcut, highcut, fs, order=6)
seegs[37:41] = 0.
seegs[52:55] = 0.

## electrodes
positions = np.load(os.path.joint(path, 'pyqt_data/positions.npy')
electrodes_color = np.loadtxt(os.path.joint(path, 'pyqt_data/electrodes_color.txt')
f = open(os.path.joint(path, 'pyqt_data/name_electrodes.txt','r')
global name_electrodes
name_electrodes = []
for line in f:
    name_electrodes.append(line)
f.close()

print('data loaded')


## QT application
app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.setWindowTitle('region simulation')
mw.resize(1000,800)
cw = QtGui.QWidget()
mw.setCentralWidget(cw)
l = QtGui.QVBoxLayout()
l = QtGui.QGridLayout()
cw.setLayout(l)


## first window
pw1 = pg.PlotWidget(name='SEEG')  
l.addWidget(pw1,0,0)
for i in range(seegs.shape[0]):
    #st = (seegs[i,:]-np.min(seegs[i,:])) / (np.max(seegs[i,:])-np.min(seegs[i,:]))
    st = seegs[i,:]
    pw1.plot(200*st+i, pen=electrodes_color[i])
lr = pg.LinearRegionItem([4000,8000])
lr2 = pg.LinearRegionItem([4000,8000])
lr.setZValue(-10)
def updatePlot1():
    indx1, indx2 = lr2.getRegion()
    indx1, indx2 = indx1, indx2
    lr.setRegion([indx1,indx2])
lr2.sigRegionChanged.connect(updatePlot1)
updatePlot1()
pw1.addItem(lr)

## second window
pw2 = pg.PlotWidget(name='Plot2', title='TAVG')
l.addWidget(pw2,1,0)
pw2.setXLink(pw1)
for i in range(tavgs.shape[0]):
    #st = (tavgs[i,:]-np.min(tavgs[i,:])) / (np.max(tavgs[i,:])-np.min(tavgs[i,:]))
    st = tavgs[i,:]
    pw2.plot(st+i, pen=(0,0,255,200))
lr2.setZValue(-10)
def updatePlot2():
    indx1, indx2 = lr.getRegion()
    indx1, indx2 = indx1, indx2
    lr2.setRegion([indx1,indx2])
lr.sigRegionChanged.connect(updatePlot2)
updatePlot2()
pw2.addItem(lr2)

# third window
pw3 = gl.GLViewWidget()
l.addWidget(pw3, 0,1)
pw3.setCameraPosition(distance=120, azimuth=-210)
mean_verts = np.mean(verts, axis=0)
max_verts = np.max(verts, axis=0)*0.01
verts = -(verts - mean_verts)/max_verts
surf_nf = faces.shape[0]
surf_nv = verts.shape[0]

surf_item = gl.GLMeshItem(vertexes=verts[:], faces=faces[:], 
                  drawFaces=True, drawEdges=False, color=(32,32,32,0.5), smooth=True,  shader='shaded')#glOptions='additive', antialias=True)

pw3.addItem(surf_item)

seeg_data = []
seeg_item = []
for i in range(seegs.shape[0]):
    seeg_data.append(gl.MeshData.sphere(rows=10, cols=10, radius=1.))
    seeg_item.append(gl.GLMeshItem(meshdata=seeg_data[i], smooth=True, shader='shaded', glOptions='additive'))
    seeg_item[i].translate(-(positions[i,0]-mean_verts[0])/max_verts[0] , -(positions[i,1]-mean_verts[1])/max_verts[1] , -(positions[i,2]-mean_verts[2])/max_verts[2] )
    pw3.addItem(seeg_item[i])
    seeg_item[i].setColor(electrodes_color[i]/255.)

centres_data1 = []
centres_item1 = []
for i in range(centres.shape[0]):
    centres_data1.append(gl.MeshData.sphere(rows=10, cols=10, radius=1.))
    centres_item1.append(gl.GLMeshItem(meshdata=centres_data1[i], smooth=True, color=(1, 0, 0, 1), shader='shaded', glOptions='additive'))
    centres_item1[i].translate(-(centres[i,0]-mean_verts[0])/max_verts[0] , -(centres[i,1]-mean_verts[1])/max_verts[1] , -(centres[i,2]-mean_verts[2])/max_verts[2] )
    pw3.addItem(centres_item1[i])


def updatePlot3():
    indx1, indx2 = lr.getRegion()
    indx1, indx2 = indx1, indx2
    tavgs_factor = 0.5
    for i in range(centres.shape[0]):
        ts_tavgs = np.sum(np.abs(tavgs[i, int(indx1):int(indx1)+10] - np.mean(tavgs[i, int(indx1):int(indx1)+10])))
        centres_item1[i].resetTransform()
        centres_item1[i].translate(-(centres[i,0]-mean_verts[0])/max_verts[0] , -(centres[i,1]-mean_verts[1])/max_verts[1] , -(centres[i,2]-mean_verts[2])/max_verts[2] )
        centres_item1[i].scale(tavgs_factor*ts_tavgs, tavgs_factor*ts_tavgs, tavgs_factor*ts_tavgs)
        centres_item1[i].meshDataChanged()

    seegs_factor = 1000
    for i in range(seegs.shape[0]):
        ts_seegs = np.sum(np.abs(seegs[i, int(indx1):int(indx1)+10] - np.mean(seegs[i, int(indx1):int(indx1)+10])))
        seeg_item[i].resetTransform()
        seeg_item[i].translate(-(positions[i,0]-mean_verts[0])/max_verts[0] , -(positions[i,1]-mean_verts[1])/max_verts[1] , -(positions[i,2]-mean_verts[2])/max_verts[2] )
        seeg_item[i].scale(1+seegs_factor*ts_seegs, 1+seegs_factor*ts_seegs, 1+seegs_factor*ts_seegs)
        seeg_item[i].meshDataChanged()

lr.sigRegionChanged.connect(updatePlot3)
updatePlot3()

## fourth window
pw4 = gl.GLViewWidget()
l.addWidget(pw4, 1,1)

pw4.setCameraPosition(distance=120, azimuth=-210)


vertcolors = np.ones((surf_nv, 4)) * np.array([0.7,0.67,0.6,0])
surf_data = gl.MeshData(vertexes=verts[:], faces=faces[:])
m1 = gl.GLMeshItem(meshdata=surf_data, smooth=True, shader='shaded')
pw4.addItem(m1)

centres_data = []
centres_item = []
for i in range(centres.shape[0]):
    centres_data.append(gl.MeshData.sphere(rows=10, cols=10, radius=1.))
    centres_item.append(gl.GLMeshItem(meshdata=centres_data[i], smooth=True, color=(1, 0, 0, 1), shader='shaded', glOptions='additive'))
    centres_item[i].translate(-(centres[i,0]-mean_verts[0])/max_verts[0] , -(centres[i,1]-mean_verts[1])/max_verts[1] , -(centres[i,2]-mean_verts[2])/max_verts[2] )
    pw4.addItem(centres_item[i])



def updatePlot4():
    indx1, indx2 = lr.getRegion()
    indx1, indx2 = indx1, indx2
    tavgs_factor = 0.5
    for i in range(centres.shape[0]):
        ts_tavgs = np.sum(np.abs(tavgs[i, int(indx1):int(indx1)+10] - np.mean(tavgs[i, int(indx1):int(indx1)+10])))
        centres_item[i].resetTransform()
        centres_item[i].translate(-(centres[i,0]-mean_verts[0])/max_verts[0] , -(centres[i,1]-mean_verts[1])/max_verts[1] , -(centres[i,2]-mean_verts[2])/max_verts[2] )
        centres_item[i].scale(2*ts_tavgs, 2*ts_tavgs, 2*ts_tavgs)

        centres_item[i].meshDataChanged()


lr.sigRegionChanged.connect(updatePlot4)
updatePlot4()

## set sizes
pw1.sizeHint = pw2.sizeHint = pw3.sizeHint = pw4.sizeHint = lambda: pg.QtCore.QSize(100, 100)
pw3.setSizePolicy(pw1.sizePolicy())
pw4.setSizePolicy(pw1.sizePolicy())

## show
mw.show()
QtGui.QApplication.instance().exec_()