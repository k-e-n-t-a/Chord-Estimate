# -*- coding: utf-8 -*-
"""
Created on Mon May 14 22:06:04 2018

@author: kuro
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display



def ChordDictionary():
    
    from collections import OrderedDict

    one_third = 1.0/3
    chord_dic = OrderedDict()
    chord_dic["C"] = [one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0,0]
    chord_dic["Db"] = [0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0,0]
    chord_dic["D"] = [0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0,0]
    chord_dic["Eb"] = [0,0,0, one_third, 0,0,0, one_third, 0,0, one_third, 0]
    chord_dic["E"] = [0,0,0,0, one_third, 0,0,0, one_third, 0,0, one_third]
    chord_dic["F"] = [one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0,0]
    chord_dic["Gb"] = [0, one_third, 0,0,0,0, one_third, 0,0,0, one_third, 0]
    chord_dic["G"] = [0,0, one_third, 0,0,0,0, one_third, 0,0,0, one_third]
    chord_dic["Ab"] = [one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0,0]
    chord_dic["A"] = [0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0,0]
    chord_dic["Bb"] = [0,0, one_third, 0,0, one_third, 0,0,0,0, one_third, 0]
    chord_dic["B"] = [0,0,0, one_third, 0,0, one_third, 0,0,0,0, one_third]
    chord_dic["Cm"] = [one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0,0]
    chord_dic["Dbm"] = [0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0,0]
    chord_dic["Dm"] = [0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0,0]
    chord_dic["Ebm"] = [0,0,0, one_third, 0,0, one_third, 0,0,0, one_third, 0]
    chord_dic["Em"] = [0,0,0,0, one_third, 0,0, one_third, 0,0,0, one_third]
    chord_dic["Fm"] = [one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0,0]
    chord_dic["Gbm"] = [0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0,0]
    chord_dic["Gm"] = [0,0, one_third, 0,0,0,0, one_third, 0,0, one_third, 0]
    chord_dic["Abm"] = [0,0,0, one_third, 0,0,0,0, one_third, 0,0, one_third]
    chord_dic["Am"] = [one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0,0]
    chord_dic["Bbm"] = [0, one_third, 0,0,0, one_third, 0,0,0,0, one_third, 0]
    chord_dic["Bm"] = [0,0, one_third, 0,0,0, one_third, 0,0,0,0, one_third]
    
    return chord_dic, len(chord_dic)


def Chromaplot(C,sr):
    
    plt.figure(figsize=(12,4))
    librosa.display.specshow(C, sr=sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
    plt.title('Chromagram')
    plt.colorbar()
    plt.tight_layout()
    #plt.savefig('./Result/test')
    plt.show()


def cos_sim(vec1,vec2):
    
    sim = np.dot(vec1,vec2) / ( np.linalg.norm(vec1) * np.linalg.norm(vec2) )

    return sim


def Momentchordestimate(C, ans, fr, dic):
    
    now_chroma = C[:,fr]
    maximum = -1

    for chord_index, (name, vector) in enumerate(dic.items()): # 0-23
        
        now_similarity = cos_sim(now_chroma, vector)
        
        if now_similarity > maximum:
            maximum = now_similarity
            now_maxchord = chord_index
            
    ans[fr] = now_maxchord
    
    return np.arange(0,fr+1,1), ans











def Chromavector(s,fs,effect='No'):
    
    
    if effect in ['yes','Yes','YES','y','Y']: # 音を分析して雑音を減らす場合
        
        # 楽音成分とパーカッシブ成分に分ける
        y, y_percussive = librosa.effects.hpss(s)
        print('雑音を除去しました')
        
    elif effect in ['no','No','NO','n','N']: # 原音をそのまま解析する場合
        
        y = s
    
    else:                                    # 原音をそのまま解析する場合
        y = s
        
    # クロマグラムを計算する
    chromavector = librosa.feature.chroma_cens(y = y, sr = fs)
    
    return chromavector
