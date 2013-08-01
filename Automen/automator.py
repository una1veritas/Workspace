import sys
import operator

if len(sys.argv) != 3 and len(sys.argv[1]) != len(sys.argv[2]) :
    print "Input parameter is wrong."
    exit

example = (sys.argv[1], sys.argv[2])
print example

substrdict = {}

for iFrom in range(0, len(example[1])) :
    for iTo in range(iFrom+1, len(example[1])) :
#        print "from %d to (before) %d" % (iFrom, iTo)
        sub = example[0][iFrom:iTo]
        label = example[1][iTo]
        substrdict[sub] = label

sortedlist = sorted(substrdict) #, key=operator.itemgetter(1, 0), reverse = False ) 
for sub in sortedlist:
    print "%s --+> %s" % (sub, substrdict[sub])
# print "text is: ", text
# worddict = {}
# 
# text = text.replace(',', ' ')
# text = text.replace('.', ' ')
# text = text.replace(':', ' ')
# text = text.replace(';', ' ')
# text = text.replace('!', ' ')
# text = text.replace('?', ' ')
# 
# words = text.split()
# 
# for word in words:
#     if word != 'I' : word = word.lower()
#     worddict[word] = worddict.get(word, 0) - 1
# 
# wordlist = worddict.iteritems()
# wordlist = sorted(wordlist, key=operator.itemgetter(1, 0), reverse = False ) 
# #itemgetter() from operator
# 
# print "Occurrences of words in the text:"
# for assoc in wordlist :
#     if len(assoc[0]) <= 6:
#         print assoc[0], '\t', -assoc[1]
#     else:
#         print assoc[0], '\n\t', -assoc[1]
# 
# # end.
