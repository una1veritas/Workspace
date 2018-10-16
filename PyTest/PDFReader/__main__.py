#
# -*- coding: utf-8 -*-

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
from glob import glob

def convert_pdf_to_txt(path): # 引数にはPDFファイルパスを指定
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    laparams.detect_vertical = True # Trueにすることで綺麗にテキストを抽出できる
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    maxpages = 0
    caching = True
    pagenos=set()
    fstr = ''
    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages,caching=caching, check_extractable=True):
        interpreter.process_page(page)

        str = retstr.getvalue()
        fstr += str

    fp.close()
    device.close()
    retstr.close()
    return fstr

def main():
    file_list = glob('../*.pdf') # PDFファイル取り込み
    
    result_list = []
    for item in file_list[:1]:
        print(item)
        result_txt = convert_pdf_to_txt(item)
        result_list.append(result_txt)
    
    allText = ','.join(result_list) # PDFごとのテキストが配列に格納されているので連結する
    
    file = open('pdf.txt', 'w')  #書き込みモードでオープン
    file.write(allText)
    
main()
