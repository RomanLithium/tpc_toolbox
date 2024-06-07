from PIL import Image
import numpy as np
import statistics as st
from sklearn import metrics
from sklearn.cluster import DBSCAN
from lecroyutils.data import LecroyScopeData as LSD
import matplotlib.pyplot as plt

######################################
#                                    # 
#   МОДУЛЬ РАБОТЫ С ОСЦИЛЛОГРАФОМ    #
#                                    #
######################################

# шаблонные имена для каналов. channel_label - номер канала
def ch_name(n, label = 1):
    if n < 10:
        s = '0000' + str(n)
    elif n < 100:
        s = '000' + str(n)
    elif n < 1000:
        s = '00' + str(n)
    elif n < 10000:
        s = '0' + str(n)
    else:
        s = str(n)
    return 'C' + str(label) + '--Trace--' + s + '.trc'

# получить x и y для осциллограммы с именем name
def wfm_unpack(name):
    tmp = LSD.parse_file(name)
    return tmp.x, tmp.y

#######################################
#                                     # 
# ПОЛУЧЕНИЕ ПЛОТНОСТИ ЭНЕРГОВЫДЕЛЕНИЯ #
#                                     #
#######################################

class track:
    
    def __init__(self):
        self.mask = [[]]
        self.base = 0
        self.brights = []
        self.ls = []
        
    # создаёт маску из самого большого кластера картинки 'maskname' (картинка, созданная методом picture.clusterme())    
    def getmask(self, maskname):
        ret = []
        arr = np.array(Image.open(maskname))
        L = len(arr)
        specter = [0]
        for o in range(L):
            for e in range(L):
                pix = arr[o, e]
                if pix not in specter:
                    specter.append(pix)
                    ret.append([[o, e]])
                elif pix != 0:
                    a = specter.index(pix)-1
                    ret[a].append([o, e])
        
        sizes = [len(z) for z in ret]
        self.mask = ret[sizes.index(max(sizes))]
        
        point = self.getstart(self.mask)
        self.ls = []
        for z in self.mask:
            self.ls.append(np.sqrt((z[0] - point[0])**2 + (z[1] - point[1])**2))
        
    # создаёт трек из картинки picname, применяя маску, созданную заранее методом .getmask().
    # Картинка должна подходить по размеру к маске
    def gettrack(self, picname):
        pic = np.array(Image.open(picname))
        
        self.base = 0
        k = 3
        for o in range(k):
            for e in range(k):
                self.base += pic[o][e]
        self.base = self.base/k/k
        
        self.brights = []
        for x in self.mask:
            self.brights.append(pic[x[0]][x[1]])
            
        self.points = self.brights
            
        xs = sorted(self.ls)
        ys = [self.brights[self.ls.index(x)] for x in xs]
        
        self.ls = xs
        self.brights = ys
            
    def getstart(self, themask):
        xs = [z[0] for z in themask]
        ys = [z[1] for z in themask]
        
        xsS = sorted(xs)
        ysS = sorted(ys)
        Lx = np.abs(xsS[-1] - xsS[0])
        Ly = np.abs(ysS[-1] - ysS[0])
        
        if Lx > Ly:
            o = xs.index(xsS[0])
        else:
            o = yx.index(ysS[0])
            
        return themask[o]
    
    # скользящее среднее по всему треку с окном n
    def shakaling(self, n):
        L = len(self.ls)
        retx = []
        rety = []
        x0 = np.max(self.ls)
        for o in range(L - n):
            retx.append(x0 - self.ls[int(o+n/2)])
            y = 0
            for e in range(n):
                y += self.brights[o+e]
            rety.append(y/n)
        self.ls = retx
        self.brights = rety
        
    # вычитает фон из трека
    def debase(self):
        self.brights = [y - self.base for y in self.brights]
        
    # выдаёт два массива - координата вдоль трека и яркость в точке
    def getgraph(self):
        return self.ls, self.brights
    
    # выдаёт маску с координатами точек и список яркостей в этих точках
    def getpoints(self):
        return self.mask, self.points

######################################
#                                    # 
# МОДУЛЬ РАБОТЫ С КАМЕРОЙ ORCA FLASH #
#                                    #
######################################

# это основной класс работы с изображениями
class picture:
    
    def __init__(self, name):
        self.im = Image.open(name)     # исходное изображение
        self.arr = np.array(self.im)   # массив, содержащий значения пикселей
        self.size = len(self.arr)      # размер изображения (сторона квадрата)
        
    # Сохраняет изображение с именем name (например, name = 'jpeg.png')
    def repack(self, name = 'picture.png'): 
        ret = self.arr.copy()
        b = np.max(ret)
        for o in range(self.size):
            for e in range(self.size):
                if ret[o][e] < 0:
                    ret[o][e] = 0
                else:
                    ret[o][e] = ret[o][e]/b*255
        Image.fromarray(np.uint8(ret)).save(name)
        
    # уменьшает размер изображения, усредняя его по клеточкам k*k пикселей
    def rescale(self, k): 
        S = self.arr.copy()
        L = self.size
        ret = []
        for o in range(int(L/k)):
            ret.append([])
            for e in range(int(L/k)):
                I = 0
                for x in range(k*o, k*o+k):
                    for y in range(k*e, k*e+k):
                        I += S[x, y]
                ret[o].append(round(I/k/k))
        
        self.size = int(L/k)
        self.arr = np.array(ret)
      
    # убивает пиксели на расстоянии меньше b от края, а также те, что ярче top и тусклее bottom
    def noisefilter_mod(self, bottom, top, b):
        he = np.max(self.arr) - np.min(self.arr) # 'высота' изображения по яркости
        for o in range(self.size):
            for e in range(self.size):
                if self.arr[o][e] < bottom*he or self.arr[o][e] > top*he or o < b or o > self.size - b - 1 or e < b or e > self.size - b - 1:
                    self.arr[o][e] = 0
                
    # превращает все ненулевые пиксели в пиксели со значением 1
    def redcloth(self):
        for o in range(self.size):
            for e in range(self.size):
                if self.arr[o][e] != 0:
                    self.arr[o][e] = 1
    
    #убивает пиксели, что ярче top и тусклее bottom
    def noisefilter(self, bottom, top):
        he = np.max(self.arr) - np.min(self.arr)
        for o in range(self.size):
            for e in range(self.size):
                if self.arr[o][e] < bottom*he or self.arr[o][e] > top*he:
                    self.arr[o][e] = 0
        
    # заменяет значение в пикселе значением медианы квадрата 3*3 с центром в этом пикселе
    def medianfilter(self):
        S = self.arr.copy()
        ret = S
        L = self.size
        for o in range(1, L-1):
            for e in range(1, L-1):
                t = []
                for x in range(o-1, o+2):
                    for y in range(e-1, e+2):
                        t.append(S[x, y])
                ret[o, e] = round(st.median(t))
                
        self.arr = ret
       
    # вычитает шум. noise_arr - массив с шумом того же размера, что и изображение
    def deletenoise(self, noise_arr):
        S = []
        for o in range(self.size):
            S.append([])
            for e in range(self.size):
                S[o].append(int(self.arr[o, e]) - int(noise_arr[o, e]))
        self.arr = np.array(S)
        
    # с этой штукой работает
    def withoutnoise(self):
        S = []
        for o in range(self.size):
            S.append([])
            for e in range(self.size):
                S[o].append(int(self.arr[o, e]))
        self.arr = np.array(S)
        
    # нормирует изображение по яркости так, чтобы самый тусклый пиксель равнялся 0, а самый яркий - 255
    def normify(self):
        zero = np.min(self.arr)
        unit = np.max(self.arr) - zero
        self.arr = (self.arr - zero)*255/unit
        print('norm = ' + str(unit))
        
    # нормирует изображение, как normify(), после чего усиливает яркость в k раз и приравнивает зашкаливающие пискели 255
    def normify_mod(self, k):
        zero = np.min(self.arr)
        unit = np.max(self.arr) - zero
        L = self.size
        for a in range(L):
            for b in range(L):
                kek = k*(self.arr[a][b] - zero)*255/unit
                if kek < 256:
                    self.arr[a][b] = kek
                else:
                    self.arr[a][b] = 255
        print('norm = ' + str(unit))
     
    # вырезает эллиптическую дырку
    def deletecenter(self, center_x = 0.477, center_y = 0.51, axis_x = 0.14, axis_y = 0.1):
        center = [center_x, center_y] # центр дырки
        radii = [axis_x, axis_y] # полуоси
        L = self.size
        for o in range(L):
            for e in range(L):
                if (o/L - center[0])**2/radii[0]**2 + (e/L - center[1])**2/radii[1]**2 < 1:
                    self.arr[o, e] = 0
                    
    # вырезает эллиптическую дырку с полупрозрачным эллиптическим ореолом вокруг
    def deletecenter2(self, center_x = 0.477, center_y = 0.51, axis_x = 0.12, axis_y = 0.08, Axis_x = 0.23, Axis_y = 0.16):
        center = [center_x, center_y] # центр дырки
        radii = [axis_x, axis_y] # полуоси
        Radii = [Axis_x, Axis_y] # полуоси ореола
        L = self.size
        for o in range(L):
            for e in range(L):
                if (o/L - center[0])**2/radii[0]**2 + (e/L - center[1])**2/radii[1]**2 < 1:
                    self.arr[o, e] = 0
                elif (o/L - center[0])**2/Radii[0]**2 + (e/L - center[1])**2/Radii[1]**2 < 1:
                    r = (o/L - center[0])**2/radii[0]**2 + (e/L - center[1])**2/radii[1]**2
                    R = (o/L - center[0])**2/Radii[0]**2 + (e/L - center[1])**2/Radii[1]**2
                    k = glad((r-1)/(r-R))
                    self.arr[o, e] = self.arr[o, e]*k
             
    # формирует и возвращает массив для всех точек изображения, состоящий из массивов [x_точки, y_точки, яркость*h/255]
    def packaspoints(self, h):
        ret = []
        for o in range(self.size):
            for e in range(self.size):
                ret.append([o, e, h*self.arr[o][e]/255])
            #print(str(o+1) + '/' + str(self.size))
        print('packed as points')
        return ret
    
    # формирует и возвращает массив из всех точек неотрицательной яркости, состоящий из массивов [x_точки, y_точки]
    def packaspoints2(self):
        ret = []
        for o in range(self.size):
            for e in range(self.size):
                if self.arr[o][e] > 0:
                    ret.append([o, e])
        print('packed as points')
        return ret
    
    # метод изображения, хранящего кластеры с треками. Возвращает длины всех кластеров, которые длиннее cut
    def Johnny(self, cut):
        labels = [0]
        xs = [[]]
        ys = [[]]
        N = self.size
        for o in range(N):
            for e in range(N):
                l = self.arr[o][e]
                if l != 0:
                    if (l not in labels):
                        labels.append(l)
                        xs.append([])
                        ys.append([])
                    xs[labels.index(l)].append(o)
                    ys[labels.index(l)].append(e)
        #print('N of tracks = ', str(len(labels) - 1))
        
        lengths = []
        for o in range(1, len(labels)):
            l = np.sqrt((np.max(xs[o]) - np.min(xs[o]))**2 + (np.max(ys[o]) - np.min(ys[o]))**2)
            if l > cut:
                lengths.append(l)
        return lengths
    
    # готовит изображение к кластеризации.
    # noisename - картинка с шумами,
    # prepname - результат подготовки,
    # stagedir - папка, куда складывать этапы подготовки,
    # stages - поставить True, чтобы сохранять этапы подготовки,
    # fingerprint - удалить эллипс в центре
    # nf_pars - параметры noisefilter_mod() в формате [bottom, top, b]
    # jackal - степень сжатия изображения
    # hole_pars - параметры отверстия в центре в формате [center_x, center_y, axis_x, axis_y, Axis_x, Axis_y] (см. deletecenter2())
    def prepare(self, noisename = 'none', prepname = 'prepared.png', stagedir = '', stages = False, fingerprint = False, jackal = 4, nf_pars = [0.15, 1.1, 2], hole_pars = [0.477, 0.51, 0.12, 0.08, 0.23, 0.16]):
        print('starting...')
        
        self.rescale(jackal)
        print('pic rescaled')

        if noisename != 'none':
            noise = picture(noisename)
            noise.rescale(jackal)
            print('noise rescaled')
        

            self.deletenoise(noise.arr)
            print('noise deleted')
            
            if stages:
                self.repack(stagedir + '1_wo_noise.png')
        else:
            self.withoutnoise()
        
        # здесь удаляется эллипс в центре. Смотреть определение deletecenter2() для настройки
        
        if fingerprint:
            center_x = hole_pars[0]
            center_y = hole_pars[1]
            axis_x   = hole_pars[2]
            axis_y   = hole_pars[3]
            Axis_x   = hole_pars[4]
            Axis_y   = hole_pars[5]
            self.deletecenter2(center_x, center_y, axis_x, axis_y, Axis_x, Axis_y)
        
        self.medianfilter()
        
        if stages:
            self.repack(stagedir + '/2_wo_center_median.png')
        
        bottom = nf_pars[0]
        top = nf_pars[1]
        b = nf_pars[2]
        self.noisefilter_mod(bottom, top, b)
        
        if stages:
            self.repack(stagedir + '/3_filtered.png')
        
        self.redcloth()
        print('pic prepared')
        
        if stages:
            self.repack(stagedir + '/4_cursed.png')
        
        self.repack(prepname)
        print('pic repacked')
        
    # кластеризует точки, складывает кластер в resultname
    # ПРИМЕНЯТЬ ТОЛЬКО ПОСЛЕ prepare()!!!
    def clusterme(self, epsilon = 15, min_smpl = 3, h = 10, starsize = 25, resultname = 'clusters.png'):
        print('############################')
        print('DBSCAN parameters:')
        print('epsilon     = ' + str(epsilon))
        print('min_samples = ' + str(min_smpl))
        print('height      = ' + str(h))
        print('############################')
        
        X = self.packaspoints2()
        db = DBSCAN(eps = epsilon, min_samples = min_smpl).fit(X)
        print('pic scanned')
        labels = db.labels_
        labels = delete_stars(labels, starsize)
        
        print('number of clusters = ' + str(len(set(labels)) - (1 if -1 in labels else 0)))
        try:
            packaspicture2(X, labels, self.size, resultname)
        except:
            print('pack failed')
            return -1
        
        print('done')
        print('#################################')
        return 0
    
###########################
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ #
###########################

# удаляет кластеры с количеством точек меньшим, чем nmax
def delete_stars(points, nmax):
    sorts = list(set(points))
    ns = [points.tolist().count(x) for x in sorts]
    ret = points
    for o in range(len(sorts)):
        for e in range(len(ret)):
            if ret[e] == sorts[o] and ns[o] <= nmax:
                ret[e] = 0
    return ret

def packaspicture2(core, fil, side, name):
    ret = [[0 for e in range(side)] for o in range(side)]
    maxx = np.max(fil)
    for o in range(len(fil)):
        if fil[o] >= 0:
            ret[core[o][0]][core[o][1]] = fil[o]/maxx*255
    Image.fromarray(np.uint8(normify(ret))).save(name)
    return 0

# нормирует массив с точками от 0 до 255
def normify(arr):
    ret = arr
    for o in range(len(ret)):
        for e in range(len(ret)):
            if ret[o][e] < 0:
                ret[o][e] = 0
    zero = np.min(arr)
    unit = np.max(arr) - zero
    print('zero = ' + str(zero))
    print('max = ' + str(unit+zero))
    print('unit = ' + str(unit))
    for o in range(len(ret)):
        for e in range(len(ret)):
            ret[o][e] = int((ret[o][e] - zero)*255/unit)
    return ret

def glad(x):
    return np.power(x, 1.3)