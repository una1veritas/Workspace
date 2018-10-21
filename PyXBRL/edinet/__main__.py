#
# -*- coding: utf-8 -*-
from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser

## init parser
parser = EdinetXbrlParser()

## parse xbrl file and get data container
xbrl_file_path = "../XBRLData/Summary/tse-qcedjpsm-78110-20181015378110-ixbrl.htm"
edinet_xbrl_object = parser.parse_file(xbrl_file_path)

print(edinet_xbrl_object)
## 例えば、該当年度の総資産を取ってみる
key = "jppfs_cor:Assets"
context_ref = "CurrentYearInstant"
current_year_assets = edinet_xbrl_object.get_data_by_context_ref(key, context_ref).get_value()

print(current_year_assets)