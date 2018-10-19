#
# -*- coding: utf-8 -*-

from glob import glob
from tabula import read_pdf

file_list = glob('../*.pdf') # PDFファイル取り込み
print(file_list)

pdf_file = file_list[0]

df = read_pdf(pdf_file)
print(df)