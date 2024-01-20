'''
Created on 2024/01/20

@author: Sin Shimozono
'''
import sys, os, glob, re
import pypdf

def main(argv):
    print(argv)
    if len(argv) > 1:
        targetdir = argv[1]
    else:
        targetdir = u"."
    dirpath = os.path.join(targetdir, u"**", u"*.pdf")
    print(dirpath)
    files = glob.glob(dirpath, recursive=True)
    basedir = os.path.dirname(__file__)
    print(str(len(files)) + " files in " + dirpath)
    
    if len(files) == 0 :
        print("no files found.")
        exit(1)
    
    writer = pypdf.PdfWriter()
    #merger = pypdf.PdfMerger()
    for pdffilename in files :
        reader = pypdf.PdfReader(pdffilename)
        # テキスト抽出
        nofpages = len(reader.pages)
        print(pdffilename, end="")
        if nofpages > 1 :
            pdfpage = reader.pages[1]
            #text = pdfpage.extract_text()
            #print(text)
        
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