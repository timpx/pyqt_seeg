from pylab import *
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
from matplotlib import cm
from scipy.signal import butter, lfilter
import os

print('load_data')
path = '/home/tim/Work/Models/python/pyqt'

# Surface
verts = np.loadtxt(os.path.join(path, 'pyqt_data/vertices.txt'))
faces = np.loadtxt(os.path.join(path, 'pyqt_data/triangles.txt'))
faces = faces.astype(int)
centres = np.loadtxt(os.path.join(path, 'pyqt_data/centres.txt'))
vertex_mapping = np.load(os.path.join(path, 'pyqt_data/vertex_mapping.npy'))
g = open(os.path.join(path, 'pyqt_data/name_regions.txt'), 'r')
global name_regions
name_regions = []
for line in g:
    name_regions.append(line)
g.close()

# TAVG
global tavgs
tavgs = np.load('../TVB/my_tvb_data/simulations/TAVG.npy')
tavgs = (np.squeeze(tavgs)).transpose()
# projection matrix and seegs
seegs_not_filtered = np.load('../TVB/my_tvb_data/simulations/TAVG_seeg.npy')
seegs_not_filtered = (np.squeeze(seegs_not_filtered)).transpose()


def butterworth_bandpass(lowcut, highcut, fs, order=5):
    """
    Build a diggital Butterworth filter

    """
    nyq = 0.5 * fs        # nyquist sampling rate
    low = lowcut / nyq    # normalize frequency
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
seegs[39] = 0

# electrodes
positions = np.loadtxt(os.path.join(path, 'pyqt_data/positions.txt'))
positions[:, 0] = -positions[:, 0]
positions[:, 1] = -positions[:, 1]


electrodes_color = np.loadtxt(os.path.join(path,
                                           'pyqt_data/electrodes_color.txt'))
f = open(os.path.join(path, 'pyqt_data/name_electrodes.txt'), 'r')
global name_electrodes
name_electrodes = []
for line in f:
    name_electrodes.append(line)
f.close()

print('data loaded')


# background color
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

# QT application
app = QtGui.QApplication([])
mw = QtGui.QMainWindow()
mw.setWindowTitle('region simulation')
mw.resize(1000, 800)
cw = QtGui.QWidget()
mw.setCentralWidget(cw)
l = QtGui.QVBoxLayout()
l = QtGui.QGridLayout()
cw.setLayout(l)


# first window
pw1 = gl.GLViewWidget()
l.addWidget(pw1, 0, 0)
lr = pg.LinearRegionItem([3999, 8000])
pw1.setCameraPosition(distance=400, azimuth=-210)
mean_verts = np.mean(verts, axis=0)
max_verts = np.max(verts, axis=0)*0.01
# verts = -(verts - mean_verts)/max_verts
surf_nf = faces.shape[0]
surf_nv = verts.shape[0]

surf_item = gl.GLMeshItem(vertexes=verts[:], faces=faces[:],
                          drawFaces=True, drawEdges=False,
                          color=(0.42, 0.42, 0.42, 1.0),
                          smooth=True, glOptions='opaque',
                          shader='edgeHilight')#, glOptions='additive')
# glOptions='additive', antialias=True)
surf_item.translate(0,0,-30)
pw1.addItem(surf_item)

seeg_data = []
seeg_item = []
for i in range(seegs.shape[0]):
    seeg_data.append(gl.MeshData.sphere(rows=10, cols=10, radius=1.))
    seeg_item.append(gl.GLMeshItem(meshdata=seeg_data[i], smooth=True,
                                   shader='shaded', glOptions='additive'))
    seeg_item[i].translate(-positions[i, 0], -positions[i, 1],
                           -positions[i, 2] - 30 )
    pw1.addItem(seeg_item[i])
    seeg_item[i].setColor(electrodes_color[i]/255.)

centres_data1 = []
centres_item1 = []
for i in range(centres.shape[0]):
    centres_data1.append(gl.MeshData.sphere(rows=10, cols=10, radius=1.))
    centres_item1.append(gl.GLMeshItem(meshdata=centres_data1[i],
                                       smooth=True, color=(1, 0, 0, 1),
                                       shader='shaded', glOptions='additive'))
    centres_item1[i].translate(centres[i, 0], centres[i, 1], centres[i, 2] - 30)
    pw1.addItem(centres_item1[i])


def updatePlot3():
    indx1, indx2 = lr.getRegion()
    indx1, indx2 = indx1, indx2
    tavgs_factor = 0.03
    tavgs_offset = 1
    for i in range(centres.shape[0]):
        ts_tavgs = np.sum(np.abs(tavgs[i, int(indx1):int(indx1)+100] -
                                 np.mean(tavgs[i, int(indx1):int(indx1)+100])))
        centres_item1[i].resetTransform()
        centres_item1[i].translate(centres[i, 0], centres[i, 1], centres[i, 2] - 30)
        centres_item1[i].scale(tavgs_offset + tavgs_factor*ts_tavgs,
                               tavgs_offset + tavgs_factor*ts_tavgs,
                               tavgs_offset + tavgs_factor*ts_tavgs)
        centres_item1[i].meshDataChanged()

    seegs_factor = 10
    for i in range(seegs.shape[0]):
        ts_seegs = np.sum(np.abs(seegs[i, int(indx1):int(indx1)+100] -
                                 np.mean(seegs[i, int(indx1):int(indx1)+100])))
        seeg_item[i].resetTransform()
        seeg_item[i].translate(positions[i, 0], positions[i, 1],
                               positions[i, 2] - 30)
        seeg_item[i].scale(1+seegs_factor*ts_seegs, 1+seegs_factor*ts_seegs,
                           1+seegs_factor*ts_seegs)
        seeg_item[i].meshDataChanged()
    # pw1.grabFrameBuffer().save('fileName.png')
lr.sigRegionChanged.connect(updatePlot3)
updatePlot3()



# second window
pw2 = pg.PlotWidget(name='SEEG_3')
l.addWidget(pw2, 1, 0)
for i in range(seegs.shape[0]):
    st = seegs[i, :]
    pw2.plot(1000*st+i, pen=electrodes_color[i])
lr.setZValue(-10)

pw2.addItem(lr)


# set sizes
pw2.sizeHint  = pw1.sizeHint = lambda: pg.QtCore.QSize(100, 100)
pw1.setSizePolicy(pw2.sizePolicy())

# show
mw.show()
QtGui.QApplication.instance().exec_()
