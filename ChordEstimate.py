# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:09:09 2018

# wav音源のコード進行を推定するプログラム
# クロマグラムを用いて推定する
# 曲のサンプリング周波数とテンポを設定する必要有り

@author: kaite kenta
"""


import librosa

filename = './Sample_AFGC.wav'

s, fs = librosa.load(filename, sr = 44100, mono = True)     #  wavファイルの読み込み
# s, fs = librosa.load(filename, sr = 44100, mono = False, offset = 5.2, duration = 10.2)

framesize = 512                                             # 1フレームの長さ
chroma = librosa.feature.chroma_cens(y= s, sr= fs, hop_length= framesize)      # クロマベクトルの算出
y_harmonic, y_percussive = librosa.effects.hpss(s)                   # 楽音成分とパーカッシブ成分に分ける
chroma = librosa.feature.chroma_cens(y= y_harmonic, sr= fs, hop_length = 512, n_octaves= 7, n_chroma= 12)



import function_CE as func

func.Chromaplot(C= chroma, sr= fs)                          # クロマグラムの表示

chord_dic, chordnum = func.ChordDictionary()                # コードの種類と数



""" コード推定 """


""" 調べるタイミング """

# 方法1. 曲のテンポと調べたい拍子間隔を設定
TEMPO = 120  
BEAT = 2

INTERVAL = 60 / TEMPO * BEAT    # コードを推定するのに使用する時間[s]

# 方法2. 検出する時間間隔を直接設定
# INTERVAL = 1

"""    end      """

import numpy as np

TONES = 12                                                  # ピッチクラス,音の種類の数
framenum = chroma.shape[1]                                  # フレーム数 1379
sum_chroma = np.zeros(TONES)                                # 判定するまでのコード点数の蓄積
TIMES = int(framenum*framesize/(INTERVAL*fs))               # 推定するコードの回数
result = np.zeros((chordnum, TIMES))                        # 最終結果(時間,コードの点数)
estimate_chords = []                                        # 最終結果のコード文字列

""" フレーム毎にコード推定する場合の設定 """

now_result = np.zeros(framenum) # now

""" """


count = 1
for frame_index in range(framenum):  #0-1378
    
    #time = frame_index * framesize # フレームの左端の時刻[s]
    #print(frame_index)
    #nth_chord = int(time/2)
    
    
    TRIG = INTERVAL * fs * count
    n = frame_index * framesize
    
    if(n <= TRIG and TRIG < n + framesize):
        print('frame=',frame_index,'TRIG=',TRIG/44100,'time=',n/fs,'error=',TRIG-n)
        
        maximum = -1
        
        for chord_index, (name, vector) in enumerate(chord_dic.items()): # 0-23
            
            similarity = func.cos_sim(sum_chroma, vector)
            #print(chord_index,int(count-1))
            result[chord_index][count-1] = similarity
            
            if( similarity > maximum):
                maximum = similarity
                this_chord = name
            
        sum_chroma = np.zeros(TONES)
        estimate_chords.append(this_chord)
        count += 1        
        
    else:
        for i in range(TONES):
            sum_chroma[i] += chroma[i][frame_index]
    
    

print(estimate_chords)


import matplotlib.pyplot as plt
import librosa.display


axis_x = np.arange(0, INTERVAL*TIMES, INTERVAL)
bar_width = 0.07
colors = ["#ff9999", "#ffaf95","#fabb92","#ffd698","#fae991","#c1fc97","#97fac8","#96f9f5","#98e1fb","#9cb2ff","#b79bfe","#fa96f9", "#b36a6a", "#ab7361","#aa7d61","#ad9165","#b4a765","#8ab66b","#6ab48f","#68b0ad","#689fb3","#6979b0","#7462a3","#aa62a9","#000000"]
for i, (name, vector) in enumerate(chord_dic.items()):
    plt.bar(axis_x - (INTERVAL * 0.45) + bar_width * i, result[i], color=colors[i], width = bar_width, label = name, align = "center")

plt.legend(loc=8,bbox_to_anchor=(1.07, 0.0))
plt.xticks(axis_x + bar_width / 25)
plt.rcParams['figure.figsize'] = (9,7)
plt.gcf().set_size_inches(9,7)
plt.xlabel('time [s]',size=20)
plt.ylabel('chord point',size=20)
plt.savefig('./test.png')
plt.show()
