'''
Created on 2024/01/18

@author: sin
'''
import sys

def collect_fileindir(dir, ext = ".pdf"):
    for filepath in glob.glob(".\**\*" + ext, recursive=True) :
        filename = os.path.splitext(os.path.basename(filepath))[0]
        print("処理開始 = %s.pdf" % filename)
        # merger = pypdf.PdfMerger()
        #
        # merger.append('data/src/pdf/sample1.pdf', pages=(0, 1))
        # merger.append('data/src/pdf/sample2.pdf', pages=(2, 4))
        # merger.merge(2, 'data/src/pdf/sample3.pdf', pages=(0, 3, 2))
        #
        # merger.write('data/temp/sample_merge_page.pdf')
        # merger.close()
            
def main(argv):
    print(len(argv))
    for each in argv :
        print(each)
        
    exit(0)

if __name__ == '__main__':
    main(sys.argv)
    