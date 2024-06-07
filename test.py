import tpc_toolbox as tpc
import matplotlib.pyplot as plt

def waveform_demo():
    # распаковать данные
    x1, y1 = tpc.wfm_unpack('data/' + tpc.ch_name(9, label = 1))
    x2, y2 = tpc.wfm_unpack('data/' + tpc.ch_name(9, label = 2))
    
    # отрисовать
    plt.plot(x1, y1, 'blue', label = 'канал 1')
    plt.plot(x2, y2, 'red', label = 'канал 2')
    plt.legend()
    plt.show()

def cluster_demo():
    # загрузить картинку
    pic = tpc.picture('data/testpic.tif')
    
    # подготовить картинку, вычтя шум и удалив центр
    pic.prepare(noisename = 'data/testnoise.png', prepname = 'data/prepared.png', stagedir = 'data/', stages = True, fingerprint = True)
    
    # кластеризовать картинку
    pic.clusterme(resultname = 'data/clusters.png')
    
def track_demo():
    trk = tpc.track()
    trk.getmask('data/clusters.png')
    trk.gettrack('data/1_wo_noise.png')
    trk.debase()
    trk.shakaling(48)
    xs, ys = trk.getgraph()
    plt.plot(xs, ys)
    plt.show()
    
waveform_demo()
cluster_demo()
track_demo()