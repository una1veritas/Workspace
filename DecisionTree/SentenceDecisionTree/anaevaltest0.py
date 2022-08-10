#coding:utf-8
#分析結果を評価する

#ファイルをオープンする
test_data = open("kranke.txt", "r")

#行ごとに全て読み込んでリストデータにする
lines = test_data.readlines()

#ファイルをクローズする
test_data.close()

#MeCabの準備
import MeCab

def mecab_list(text):
  tagger = MeCab.Tagger("-Ochasen")
  tagger.parse('')
  node = tagger.parseToNode(text)
  word_class = []
  while node:
      word = node.surface
      wclass = node.feature.split(',')
      if wclass[0] != u'BOS/EOS':
          if wclass[6] == None:
              #word_class.append(list((list(word),list(wclass[0]),list(wclass[1]))))#,wclass[2],"")))
              word_class.append(list((word,wclass[0],wclass[1])))
          else:
              #word_class.append(list((list(word),list(wclass[0]),list(wclass[1]))))#,wclass[2])))#,wclass[6])))
              word_class.append(list((word,wclass[0],wclass[1])))
      node = node.next
  return word_class

#分かち書きした結果を表示

#全て表示
#for line in lines:
#    test = mecab_list(line)
#    print(test)

#analist = list()
#for line in lines:
#    analist.append(mecab_list(line))
#    test = mecab_list(line)
#    if test[0][2] == '一般':
#        print(test)
#print(analist)

#フラグたてした部分の取り出し
#for line in lines:
#    test = []
#    test = mecab_list(line)
#    elem = len(test)
#    for line in lines:
#    test = mecab_list(line)
#    for num in range(elem):
#        if test[num][0] == 'True':
#            print(test[0])

#
#なんか前やった決定木の作成は答えを目指すものだったけど、
#今回答えがないので分類には別の関数使うのかなとか思ってしまう。
#

#解析結果をリストに変換
ana_list = list()
for line in lines:
    mov = mecab_list(line)
    ana_list.append(mov)
#print(ana_list)



#ここから決定木
#from sklearn.datasets import load_iris
#from sklearn import tree

#要素リストより、要素が入った集合を作る
wordset = set() #単語の集合
numb = len(ana_list) #文章数
for i in range(numb):
    numt = len(ana_list[i])
    for j in range(numt-1):
        elem_tuple  = tuple(ana_list[i][j])
        wordset.add(elem_tuple)

print(wordset)


#クエリとしての評価値を返す関数


def mygini(ana_list,wordset,qword):
    tcount = 0  #True文の数
    for j in range(len(ana_list)):
        if ana_list[j][len(ana_list[j])-1][0] == 'True':
            tcount += 1
    fcount = len(ana_list) - tcount #False文の数
    if tcount == 0 or fcount == 0:
        print(qword)
        print(ana_list)
        return
    min_gini = 1
    mg_list = list()
    for word in wordset:
        wordl = list(word)
        ttcount = 0
        ftcount = 0
#
#正常に動く部分
#
        for k in range(len(ana_list)):
            if ana_list[k][len(ana_list[k])-1][0] == 'True':
                for l in range(len(ana_list[k])):
                    if wordl == ana_list[k][l]:
                        ttcount += 1
                        break
            else:
                for m in range(len(ana_list[k])):
                    if wordl == ana_list[k][m]:
                        ftcount += 1
                        break

#
#
#

        #gini係数を求める計算
        tfcount = tcount - ttcount
        ffcount = fcount - ftcount
        #文がTrueのジニ係数
        t_gini = 1 - (ttcount/tcount)**2 - (tfcount/tcount)**2
        #文がFalseのジニ係数
        f_gini = 1 - (ftcount/fcount)**2 - (ffcount/fcount)**2
        #ジニ係数
        total_gini = (tcount/len(ana_list))*t_gini + (fcount/len(ana_list))*f_gini
#        print(wordl[0] + "のジニ係数=" + str(total_gini))
        if total_gini < min_gini:
            min_gini = total_gini
            mg_list = [min_gini,wordl]
#    print(mg_list)
    #setのwordからwordを抜く
    true_list = list()
    false_list = list()
    print('--------------------------------------------------------------------------')


#
#エラー部分
#

    for poi in range(0,len(ana_list)):
        #print(ana_list[poi])
        #print(poi)
        for oi in range(0,len(ana_list[poi])):
            #print(len(ana_list[poi][oi]))
            if mg_list[1] == ana_list[poi][oi]:
                true_list.append(ana_list[poi])
                ana_list.remove(ana_list[poi])

#
#
#

    wordlist = list(wordset)
    wordlist.remove(word)
    wordset = set(wordlist)
    false_list = ana_list
    wordw = wordl[0]
    mygini(true_list,wordset,wordw)
    mygini(false_list,wordset,wordw)
    print(mg_list)

mygini(ana_list,wordset,'start')






#target = [u[0][2] for u in ana_list]   #結果リスト
#for line in lines:
#    test = []
#    test = mecab_list(line)
#    elem = len(test)
#    test = data.append(mecab_list(line))
#target = [u[3] for u in table]  #目的変数（単位の有無）を抽出
#clf = tree.DecisionTreeClassifier() #インスタンスを生成

#clf = clf.fit(data,target)  #データで学習させる
#predicted = clf.predict(data) #予測を実行

#print(sum(predicted == target) / len(target))


#import pydotplus
#from sklearn.externals.six import StringIO
#dot_data = StringIO()
#tree.export_graphviz(clf,out_file=dot_data)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("krankeanagraph.pdf")
