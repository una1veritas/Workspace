import operator

textfile = open('KR2a2s1.txt')
try:
    text = textfile.read();
finally:
    textfile.close()

print "text is: ", text
worddict = {}

text = text.replace(',', ' ')
text = text.replace('.', ' ')
text = text.replace(':', ' ')
text = text.replace(';', ' ')
text = text.replace('!', ' ')
text = text.replace('?', ' ')

words = text.split()

for word in words:
    if word != 'I' : word = word.lower()
    worddict[word] = worddict.get(word, 0) - 1

wordlist = worddict.iteritems()
wordlist = sorted(wordlist, key=operator.itemgetter(1, 0), reverse = False ) 
#itemgetter() from operator

print "Occurrences of words in the text:"
for assoc in wordlist :
    if len(assoc[0]) <= 6:
        print assoc[0], '\t', -assoc[1]
    else:
        print assoc[0], '\n\t', -assoc[1]

# end.
