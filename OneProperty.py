import tensorflow as tf
import numpy as np
import IDS_Data as ID
import random
import IntrusiveCNN as CNN


class Disturb():
    def __init__(self, ORX=0, ORY=0, perturbation=0):
        self.ORX=ORX
        self.ORY=ORY
        self.perturbation=perturbation

x,y=ID.get_next_batch(1)
x,y=ID.get_next_batch(1)
print(x.shape)

def DE():
    entropy_list=[]
    x_copy=x.copy()
    selected_per=None
    selected_entropy=0
    perturbation=[]
    for i in range(400):
        x_copy=x.copy()
        perturbation_i=Disturb(random.randint(0,10),random.randint(0,11),float(-(random.randint(0,10))/10))
        perturbation.append(perturbation_i)
        x_copy[0,perturbation_i.ORX,perturbation_i.ORY,0]=x[0,perturbation_i.ORX,perturbation_i.ORY,0]+perturbation_i.perturbation
        entropy_list.append(CNN.inference(x_copy,y))
        print('Times:',i,':',perturbation[i].ORX,perturbation[i].ORY,perturbation[i].perturbation,'entropy:',entropy_list[i])
    print('original entropy:',CNN.inference(x,y))
    print('entropy_list:',entropy_list)
    
    for ite in range(100):
        for i in range(400):
            x_copy=x.copy()
            #x_copy=x
            print('x_id:',id(x),'x_copy_id:',id(x_copy))
            r1=random.randint(0,399)
            r2=random.randint(0,399)
            r3=random.randint(0,399)
            s1=random.randint(0,399)
            s2=random.randint(0,399)
            s3=random.randint(0,399)
            while(r1==r2 or r2==r3 or r1==r3):
                r1=random.randint(0,399)
                r2=random.randint(0,399)
                r3=random.randint(0,399)
            while(s1==s2 or s2==s3 or s1==s3):
                s1=random.randint(0,399)
                s2=random.randint(0,399)
                s3=random.randint(0,399)
            pertubatiob_in_i=Disturb(int(perturbation[r1].ORX+0.5*(perturbation[r2].ORX+perturbation[r3].ORX))%11,int(perturbation[r1].ORY+0.5*(perturbation[r2].ORY+perturbation[r3].ORY))%12,perturbation[r1].perturbation+0.5*(perturbation[r2].perturbation+perturbation[r3].perturbation))
            #pertubatiob_in_j=Disturb(int(perturbation[s1].ORX+0.5*(perturbation[s2].ORX+perturbation[s3].ORX))%11,int(perturbation[s1].ORY+0.5*(perturbation[s2].ORY+perturbation[s3].ORY))%12,perturbation[s1].perturbation+0.5*(perturbation[s2].perturbation+perturbation[s3].perturbation))
            x_copy[0,perturbation[i].ORX,perturbation[i].ORY,0]=x[0,pertubatiob_in_i.ORX,pertubatiob_in_i.ORY,0]+pertubatiob_in_i.perturbation
            entropy_in_i=CNN.inference(x_copy,y)
            entropy_ori=entropy_list[i]
            if (entropy_in_i>entropy_ori and -2<pertubatiob_in_i.perturbation<2):
                entropy_list[i]=entropy_in_i
                perturbation[i]=pertubatiob_in_i
            print('Epoches:',ite,'Times:',i,':',perturbation[i].ORX,perturbation[i].ORY,perturbation[i].perturbation,'entropy:',entropy_list[i],'per:',perturbation[r1].perturbation+0.5*(perturbation[r2].perturbation+perturbation[r3].perturbation))
            if (entropy_list[i]>1.5):
                return
