#coding:utf-8
#MeCabの準備
import MeCab
#分析結果を評価する

def mecab_list(text):
    tagger = MeCab.Tagger("-Ochasen")
#    tagger.parse('')
    node = tagger.parseToNode(text)
    word_class = []
    while node:
        word = node.surface
        wclass = node.feature.split(',')
        if wclass[0] != u'BOS/EOS':
#            if wclass[6] == None:
                # word_class.append(list((list(word),list(wclass[0]),list(wclass[1]))))#,wclass[2],"")))
                word_class.append(list((word, wclass[0], wclass[1])))
#            else:
                # word_class.append(list((list(word),list(wclass[0]),list(wclass[1]))))#,wclass[2])))#,wclass[6])))
#                word_class.append(list((word, wclass[0], wclass[1])))
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

#ファイルをオープンする
data = list()
with open("kranke.csv", "r") as textfile:
    #行ごとに全て読み込んでリストデータにする
    for line in textfile.readlines():
        data.append([item.strip() for item in line.split(',')])
#ファイルをクローズする
#test_data.close()
#print(data)

tagger = MeCab.Tagger("-Ochasen")
for record in data :
    node = tagger.parseToNode(record[0])
    parsed = list()
    while node:
        winfo = node.feature.split(',')
        if winfo[0] != u'BOS/EOS':
            parsed.append( (node.surface, winfo[0], winfo[1]) )
        node = node.next
    record.append(parsed)

print(data)

words = dict()
for a_record in data :
    for word in a_record[2] :
        if (word[1], word[2]) in [('助詞', '格助詞'), ('助詞', '接続助詞'), ('助動詞', '*')]:
            continue
        if (word[1], word[2]) not in words :
            words[(word[1], word[2])] = set()
        words[(word[1], word[2])].add(word[0])

for key, value in words.items():
    print(key, value)
#解析結果をリストに変換
#ana_list = list()
#for line in table:
#    mov = mecab_list(line[0])
#    ana_list.append(mov)
#print(ana_list)



#ここから決定木
#from sklearn.datasets import load_iris
#from sklearn import tree

#要素リストより、要素が入った集合を作る
#wordset = set() #単語の集合
#numb = len(ana_list) #文章数
#for i in range(numb):
#    numt = len(ana_list[i])
#    for j in range(numt-1):
#        elem_tuple  = tuple(ana_list[i][j])
##
#pprint.pprint(wordset)
#pprint.pprint(ana_list)

wordset = set()
for ws in words.values():
    wordset |= ws
print(wordset)
ana_list = data
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
    if len(mg_list) != 0:
        delnum = list()
        #print(mg_list)
        #while poi < len
        for poi in range(len(ana_list)):
            #print(ana_list[poi])
            #print(poi)
            for oi in range(len(ana_list[poi])):
                #print(ana_list[poi][oi])
                #print(oi)
                if mg_list[1] == ana_list[poi][oi]:
                    true_list.append(ana_list[poi])
                    remword = set() #単語の集合
                    delnum.append(ana_list[poi])
                    #ana_list.remove(ana_list[poi])
                    break

        for remnumb in range(len(true_list)):
            for remnumt in range(len(true_list[remnumb])-1):
                relem_tuple  = tuple(true_list[remnumb][remnumt])
                if mg_list[1] != list(relem_tuple) and relem_tuple != wordset:
                    remword.add(relem_tuple)

        noi = 0
        while noi < len(delnum):
            ana_list.remove(delnum[noi])
            noi+=1

#
#
#
        wordlist = list(wordset)
        wordt = tuple(mg_list[1])
        wordlist.remove(wordt)
        false_list = ana_list
        if len(remword) != 0:
            for wordr in remword:
                remfl = 0
                wordrl = list(wordr)
                for fanumb in range(len(false_list)):
                    for fanumt in range(len(false_list[fanumb])):
                        if wordrl == false_list[fanumb][fanumt]:
                            remfl = 1
                if remfl == 0:
                    wordlist.remove(tuple(wordrl))
        wordw = mg_list[1][0]
        wordset = set(wordlist)
        mygini(true_list,wordset,wordw)
        mygini(false_list,wordset,wordw)

        return
#        print(mg_list)
    #else:
        #print(qword)
        #print(ana_list)

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
