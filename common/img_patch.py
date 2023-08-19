import numpy as np 
import skimage.transform as trans
import math

def patch_Gen(self,target_size=(256,256), flag_multi_class=False):
    sx=self.shape[0]
    sy=self.shape[1]
    x0=math.floor((512-divmod(sx,512)[1])/divmod(sx,512)[0])
    y0=math.floor((512-divmod(sy,512)[1])/divmod(sy,512)[0])
    for i in (range(divmod(sx,512)[0]+1)):
        for k in (range(divmod(sy,512)[0]+1)):
            dx=(512-int(x0))*i
            dy=(512-int(y0))*k
            img=self[dx:dx+512,dy:dy+512]
            #img = img / 255   # すでに０−１のコントラストになっているのでこの項目やめる
            img = trans.resize(img,target_size)
            img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
            img = np.reshape(img,(1,)+img.shape)
            yield img

def patch_Counter(self):
    sx=self.shape[0]
    sy=self.shape[1]
    x0=math.floor((512-divmod(sx,512)[1])/divmod(sx,512)[0])
    y0=math.floor((512-divmod(sy,512)[1])/divmod(sy,512)[0])
    n=0
    pos=[]
    for i in (range(divmod(sx,512)[0]+1)):
        for k in (range(divmod(sy,512)[0]+1)):
            n=n+1
            pos.append([n, (512-int(x0))*i, (512-int(y0))*k])
    return pos

def patch_Assemble(self, results):
    pos=patch_Counter(self)
    sx=self.shape[0]+5
    sy=self.shape[1]+5
    img=np.zeros((sx,sy))
    for i in range(len(pos)):
            img[pos[i][1]:512+pos[i][1], pos[i][2]:512+pos[i][2]] = trans.resize(results[i][:,:,0],(512,512))
    return img
