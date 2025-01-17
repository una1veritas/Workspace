'''
Created on 2025/01/15

@author: sin
'''

FEEDBACK_XML_HEADER_ITEMS = '''<?xml version="1.0" encoding="UTF-8" ?>
<FEEDBACK VERSION="200701" COMMENT="XML-Importfile for mod/feedback">
     <ITEMS>
'''
FEEDBACK_XML_FOOTER_ITEMS = '''     </ITEMS>
</FEEDBACK>
'''
FEEDBACK_ITEM_1ST_MULTICHOICE = '''          <ITEM TYPE="multichoice" REQUIRED="1">
               <ITEMID>
                    <![CDATA[{item_id}]]>
               </ITEMID>
               <ITEMTEXT>
                    <![CDATA[第1希望の研究室を選択してください。]]>
               </ITEMTEXT>
               <ITEMLABEL>
                    <![CDATA[]]>
               </ITEMLABEL>
               <PRESENTATION>
                    <![CDATA[d>>>>>{choice_rows}]]>
               </PRESENTATION>
               <OPTIONS>
                    <![CDATA[h]]>
               </OPTIONS>
               <DEPENDITEM>
                    <![CDATA[0]]>
               </DEPENDITEM>
               <DEPENDVALUE>
                    <![CDATA[]]>
               </DEPENDVALUE>
          </ITEM>'''
FEEDBACK_ITEM_1ST_EXTENDED_MULTICHOICE = '''          <ITEM TYPE="multichoice" REQUIRED="0">
               <ITEMID>
                    <![CDATA[{item_id}]]>
               </ITEMID>
               <ITEMTEXT>
                    <![CDATA[第1希望の研究室で「学科外の研究室」を選択した場合は、第1希望の学科外の研究室を選択してください。]]>
               </ITEMTEXT>
               <ITEMLABEL>
                    <![CDATA[]]>
               </ITEMLABEL>
               <PRESENTATION>
                    <![CDATA[d>>>>>{choice_rows}]]>
               </PRESENTATION>
               <OPTIONS>
                    <![CDATA[h]]>
               </OPTIONS>
               <DEPENDITEM>
                    <![CDATA[0]]>
               </DEPENDITEM>
               <DEPENDVALUE>
                    <![CDATA[]]>
               </DEPENDVALUE>
          </ITEM>
'''
FEEDBACK_ITEM_ITH_MULTICHOICE = '''          <ITEM TYPE="multichoice" REQUIRED="{flag_required}">
               <ITEMID>
                    <![CDATA[{item_id}]]>
               </ITEMID>
               <ITEMTEXT>
                    <![CDATA[第{desired_rank}希望の研究室を選択してください。]]>
               </ITEMTEXT>
               <ITEMLABEL>
                    <![CDATA[]]>
               </ITEMLABEL>
               <PRESENTATION>
                    <![CDATA[d>>>>>{choice_rows}]]>
               </PRESENTATION>
               <OPTIONS>
                    <![CDATA[h]]>
               </OPTIONS>
               <DEPENDITEM>
                    <![CDATA[0]]>
               </DEPENDITEM>
               <DEPENDVALUE>
                    <![CDATA[]]>
               </DEPENDVALUE>
          </ITEM>
'''
item_id = 22708
multichoice_choices = ['''01：坂本 比呂志
|02：佐藤 好久
|03：宮野 英次
|04：井 智弘
|05：大北 剛
|06：斎藤 寿樹
|07：下薗 真一
|08：徳永 旭将
|09：藤本 晶子
|10：本田 あおい
|11：乃美 正哉
|12：嶋田 和孝
|13：平田 耕一
|14：國近 秀信
|15：中村 貞吾
|16：尾下 真樹
|17：齊藤 剛史
|18：武村 紀子
|19：新見 道治
|20：学科外の研究室''',
'''31：馬場 昭好（マイクロ化総合技術センター）
|32：中村 和之（マイクロ化総合技術センター）
|33：山田 雅之（教養教育院）
|35：玉川 雅章（生命体工学研究科）
|36：安田 隆（生命体工学研究科）
|37：山田 宏（生命体工学研究科）
|38：高嶋 一登（生命体工学研究科）
|39：春山 哲也（生命体工学研究科）
|40：池野 慎也（生命体工学研究科）
|41：加藤 珠樹（生命体工学研究科）
|42：前田 憲成（生命体工学研究科）
|43：安藤 義人（生命体工学研究科）
|44：久米村 百子（生命体工学研究科）
|46：渡邊 晃彦（生命体工学研究科）
|47：宮崎 敏樹（生命体工学研究科）
|48：中村 仁（生命体工学研究科）
|49：高辻 義行（生命体工学研究科）
|50：本田 英己（生命体工学研究科）
|51：石井 和男（生命体工学研究科）
|52：田中 啓文（生命体工学研究科）
|53：和田 親宗（生命体工学研究科）
|55：田向 権（生命体工学研究科）
|56：古川 徹生（生命体工学研究科）
|57：柴田 智広（生命体工学研究科）
|58：堀尾 恵ー（生命体工学研究科）
|59：我妻 広明（生命体工学研究科）
|60：吉田 香（生命体工学研究科）
|61：夏目 季代久（生命体工学研究科）
|62：立野 勝巳（生命体工学研究科）
|63：大坪 義孝（生命体工学研究科）
|64：井上 創造（生命体工学研究科）
|65：安川 真輔（生命体工学研究科）
|66：池本 周平（生命体工学研究科）
|67：西田 祐也（生命体工学研究科）
|68：田中 悠一朗（生命体工学研究科）
|69：常木 澄人（生命体工学研究科）''']

def get_table_from_csv(filename, headers = False) :
    tbl = list()
    column_headers = None
    with open(filename, "r", encoding="utf-8-sig") as f:            
        rowcount = 0
        for a_line in f:
            a_line = a_line.strip()
            if len(a_line) == 0 : 
                continue
            a_row = a_line.split(',')
            if all([len(c) == 0 for c in a_row]) :
                continue
            rowcount += 1
            if rowcount == 1 and headers :
                column_headers = [ h for h in a_row]
                continue
            tbl.append(a_row)
    if column_headers != None :
        return (column_headers, tbl)
    else:
        return tbl

if __name__ == '__main__':
    headers, supervisors = get_table_from_csv('2025年度卒研指導教員.csv', True)
    # for l in supervisors:
    #     print(l)
    multichoice_ai = ''
    for row in supervisors:
        # print(row)
        if row[2].startswith('知能情報') or row[2].startswith('学科外') :
            if row[3].startswith('配属なし') :
                continue
            if len(multichoice_ai) > 0 :
                multichoice_ai += '\n|'
            multichoice_ai += '{0:02}：{1}'.format(int(row[0]), row[1])
    multichoice_nonai = ''
    for row in supervisors:
        # print(row)
        if not row[2].startswith('知能情報') and not row[2].startswith('学科外')  :
            if row[3].startswith('配属なし') :
                continue
            if len(multichoice_nonai) > 0 :
                multichoice_nonai += '\n|'
            multichoice_nonai += '{0:02}：{1}'.format(int(row[0]), row[1])
    # print(multichoice_nonai)
    idnum = 22700
    print(FEEDBACK_XML_HEADER_ITEMS)
    idnum += 1
    print(FEEDBACK_ITEM_1ST_MULTICHOICE.format(item_id=idnum, choice_rows=multichoice_ai))
    idnum += 1
    print(FEEDBACK_ITEM_1ST_EXTENDED_MULTICHOICE.format(item_id=idnum, choice_rows=multichoice_nonai))
    for i in range(2,20):
        idnum += 1
        if i <= 6 :
            flg = '1'
        else:
            flg = '0'
        print(FEEDBACK_ITEM_ITH_MULTICHOICE.format(flag_required=flg, item_id=idnum, desired_rank=i, choice_rows=multichoice_ai))
    print(FEEDBACK_XML_FOOTER_ITEMS)
