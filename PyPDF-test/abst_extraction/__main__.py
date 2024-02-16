'''
Created on 2024/01/20

@author: Sin Shimozono
'''
import sys, os, glob, re
import pypdf
from operator import itemgetter

registered_name = {'中村貞吾', '齊藤剛史', '下薗真一', 
                   '古川徹生', '徳永旭将', '大北剛', '佐藤好久', 
                   '乃美正哉', '新見道治', '武村紀子', '井智弘', 
                   '硴崎賢一', '坂本⽐呂志', '二反田篤史', '乃万司', 
                   '岡部孝弘', '宮野英次', '田向権', '國近秀信', 
                   '平田耕一', '藤本晶子', '嶋田和孝', '斎藤寿樹', 
                   '尾下真樹', '本田あおい', '和田親宗'}

def nearestapprox(name, registered):
    if '研究室' in name :
        name = name.split('研究室')[0]
    if name in registered :
        return name
    best = (0, "")
    for each in registered :
        count = 0
        for pair in zip(name, each):
            if (pair[0] != pair[1]) : break
            count += 1
        if count > best[0] : 
            best = (count, each)
    return best[1]

def main(argv):
    progbasedir = os.path.dirname(__file__)
    print(argv, progbasedir)
    if len(argv) > 1:
        targetdir = argv[1]
    else:
        targetdir = u"."
    dirpath = os.path.join(targetdir, u"**", u"*.pdf")
    print(dirpath)
    paths = glob.glob(dirpath, recursive=True)
    if len(paths) == 0 :
        print("no pdf files found.")
        exit(1)
    
    db = list()
    for filepath in paths :
        filename = os.path.basename(filepath)
        parts = os.path.splitext(filename)[0].split('_')
        if len(parts) > 3 :
            print("warning: too many _'s in file name: " + str(parts))
        (supervisor, sid, sname) = (parts[0], parts[1], parts[2])
        supervisor = nearestapprox(supervisor, registered_name)
        db.append([sid, sname, supervisor, filepath])
        #print(db[-1])
    db.sort(key = itemgetter(0))
    db.sort(key = itemgetter(1))
    for entry in db :
        print(entry)
    print(str(len(paths)) + " files in " + dirpath)
    
#    exit(0)
    
    writer = pypdf.PdfWriter()
    #merger = pypdf.PdfMerger()
    for (sid, sname, supervisor, pdffilename) in db :
        reader = pypdf.PdfReader(pdffilename)
        # テキスト抽出
        nofpages = len(reader.pages)
        print(pdffilename, end="")
        if nofpages > 1 :
            # titlepagetext = reader.pages[0].extract_text()
            # print(titlepagetext)
            pdfpage = reader.pages[1]
            writer.add_page(pdfpage)
            print(", " + str(nofpages) + "pages.", end="")
        else:
            print()
            print("error!! The document has insufficient pages.")
        print() 
    writer.write("sampleout.pdf")
    print("bye.")

if __name__ == '__main__':
    main(sys.argv)