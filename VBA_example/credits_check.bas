Attribute VB_Name = "Module1"
'Option Explicit

Type KeyValuePair
    key As String
    value As Variant
End Type

Public Function GetFilename(fname As String, msg As String)
    If fname <> "" Then
        fname = Dir(fname)
    End If
    If fname = "" Then
        fname = Application.GetOpenFilename("xlsxƒtƒ@ƒCƒ‹(*.xlsx),*xlsx", 1, msg)
    End If
    GetFilename = fname
End Function

Public Function LastRow(wks As Variant, col As Integer) As Integer
    LastRow = wks.Cells(wks.Rows.Count, col).End(xlUp).row    'col —ñ–Ú‚ÅÅIs‚ğ’T‚·
End Function

Public Function RightmostColumn(wks As Worksheet, r As Integer) As Integer
    RightmostColumn = wks.Cells(r, wks.Columns.Count).End(xlToLeft).Column    'row s–Ú‚ÅÅI—ñ‚ğ’T‚·
End Function

Sub ƒtƒ@ƒCƒ‹İ’è()
    Dim wkst As Worksheet
    Set wkst = ThisWorkbook.Sheets("ƒRƒ“ƒgƒ[ƒ‹ƒpƒlƒ‹")
    txt = GetFilename(wkst.Range("B2"), wkst.Range("A2").value & "‚ğŠJ‚­")
    wkst.Range("B2").value = txt
    txt = GetFilename(wkst.Range("B4"), wkst.Range("A4").value & "‚ğŠJ‚­")
    wkst.Range("B4").value = txt
    wkst.Range("B5").value = ThisWorkbook.Path
End Sub


Sub ¬Ñ•sUƒ`ƒFƒbƒN()
    Dim WsMain As Worksheet
    Dim WsRecords As Worksheet, WsSummary As Worksheet
    
    Dim Records As Variant
    Dim Summay As Variant
    
    Dim rangeTopLeft As Range  ' •\‚Ì¶ã‹÷ƒZƒ‹‚ğw‚·”ÍˆÍ
    
    Set WsMain = ThisWorkbook.Sheets("ƒRƒ“ƒgƒ[ƒ‹ƒpƒlƒ‹")
    
    Call ƒtƒ@ƒCƒ‹İ’è
    
    '’PˆÊC“¾ó‹µƒ[ƒNƒV[ƒg
    Set WsSummary = Workbooks.Open(WsMain.Range("B2").value).Worksheets(1)
    
    'ŠwĞˆÚ“®‹L˜^ƒ[ƒNƒV[ƒg
    Set WsRecords = Workbooks.Open(WsMain.Range("B4").value).Worksheets(1)
    
    'ƒRƒ“ƒgƒ[ƒ‹ƒpƒlƒ‹ƒ[ƒNƒV[ƒg‚Ìî•ñ‚ÉŠî‚Ã‚­ŠwĞˆÙ“®ƒCƒxƒ“ƒg‚ÌûW
    Dim RColSID As String  'Šw¶”Ô†Œ©o‚µ—ñ
    Dim RColBDate As String  'ˆÙ“®ŠJnŒ©o‚µ—ñ
    Dim RColEDate As String  'ˆÙ“®I—¹Œ©o‚µ—ñ
    Dim RColTrans As String  'ˆÙ“®“à—eŒ©o‚µ—ñ
    Dim RecordsCount As Integer
                                            
    Set rangeTopLeft = WsRecords.Range(WsMain.Range("C4").value)  '—ñŒ©o‚µ¶’[ƒZƒ‹‚ğ”ÍˆÍ‚É
    Debug.Print "rangeTopLeft is ", VarType(rangeTopLeft), rangeTopLeft
    
    bottomRow = LastRow(WsRecords, rangeTopLeft.Column)  '1—ñ–Ú‚Ås”‚ğƒJƒEƒ“ƒg
    RecordsCount = bottomRow - rangeTopLeft.row          '‘ƒCƒxƒ“ƒg”iƒf[ƒ^s”j
    'Debug.Print rangeTopLeft.Address, rangeTopLeft.Column, rangeTopLeft.Row, bottomRow, RecordsCount

    For cx = 1 To RightmostColumn(WsRecords, rangeTopLeft.row)
        If rangeTopLeft.Offset(0, cx).value = "Šw¶”Ô†" Then
            RColSID = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("D4").value = RColSID
        ElseIf rangeTopLeft.Offset(0, cx).value = "ˆÙ“®ŠJn" Then
            RColBDate = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("E4").value = RColBDate
        ElseIf rangeTopLeft.Offset(0, cx) = "ˆÙ“®I—¹" Then
            RColEDate = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("F4").value = RColEDate
        ElseIf rangeTopLeft.Offset(0, cx) = "ˆÙ“®“à—e" Then
            RColTrans = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("G4").value = RColTrans
        End If
    Next cx

    ' ŠwĞˆÚ“®ƒCƒxƒ“ƒg‚ğiŠw¶”Ô†AˆÚ“®‚ª¶‚¶‚½“ú•tj‚Ì‡‚Åƒ\[ƒg
    WsRecords.Range(rangeTopLeft.Address).Resize(RecordsCount + 1, 15) _
        .Sort Key1:=WsRecords.Range(RColSID), Key2:=WsRecords.Range(RColBDate), Header:=xlYes
    '”z—ñ Records ‚ÉŠi”[iƒf[ƒ^•”‚Ì‚İj
    ReDim Records(RecordsCount + 1, 4)
    '”O‚Ì‚½‚ß“ú•t‚ğ vbDate ‚É‘µ‚¦‚é
    With WsRecords
        For rx = 1 To RecordsCount
            '‹xŠwˆÈŠO‚à‚Æ‚è‚±‚İ
            Records(rx, 1) = .Range(RColSID).Offset(rx, 0).value
            Records(rx, 2) = CDate(.Range(RColBDate).Offset(rx, 0).value)
            EndDate = .Range(RColEDate).Offset(rx, 0).value
            If EndDate <> "" Then
                Records(rx, 3) = CDate(EndDate)
            Else
                Records(rx, 3) = ""
            End If
            Records(rx, 4) = .Range(RColTrans).Offset(rx, 0).value
        Next rx
    End With
    'ReDim Records(RecordsCount + 1, 4)
    
    Debug.Print UBound(Records, 2)
    Debug.Print "Records = "
    For rx = 1 To UBound(Records, 1)
        If Records(rx, 1) = -1 Then Exit For
        For cx = 1 To UBound(Records, 2)
            Debug.Print Records(rx, cx),
        Next cx
        Debug.Print
    Next rx
    Debug.Print
    'Stop

    '’PˆÊC“¾ó‹µ‚ğŠw¶”Ô†‡‚Åƒ\[ƒg
    Set rangeTopLeft = WsSummary.Range(WsMain.Range("C2").value)  'Šw¶”Ô†A’PˆÊ‚È‚Ç‚ÌŒ©o‚µsÅ¶’[
    bottomRow = LastRow(WsSummary, rangeTopLeft.Column)  '1—ñ–Ú‚Ås”‚ğƒJƒEƒ“ƒg
    rightColumn = RightmostColumn(WsSummary, rangeTopLeft.row)
    
    SIDsCount = bottomRow - rangeTopLeft.row               'Šw¶”Ô†”
    semesCount = rightColumn - rangeTopLeft.Column
    
    WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, semesCount + 1).Sort Key1:=rangeTopLeft, Header:=xlYes
    Summary = WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, semesCount + 1).value
    'WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, SemesCount + 1).Select
    'Debug.Print SIDsCount, SemesCount
    
    '’PˆÊC“¾ó‹µ•\‚ÌŒ©o‚µ—ñi”N“xŠwŠúj‚ğæ“¾
    With WsSummary.Range(rangeTopLeft.Address).Offset(-2, 1).Resize(2, UBound(Summary, 2) - 1)
        For cx = 1 To .Columns.Count
            '2022-10 Œ`®
            s = .Cells(1, cx).MergeArea(1, 1).value
            If .Cells(2, cx).value = "‘OŠú" Then
                Summary(1, 1 + cx) = s & "-04"
            ElseIf .Cells(2, cx).value = "ŒãŠú" Then
                Summary(1, 1 + cx) = s & "-10"
            Else
                Summary(1, 1 + cx) = s & "-ƒGƒ‰[ŠwŠú"
            End If
        Next cx
    End With
    WsMain.Range("D2").value = Summary(1, 2)
    WsMain.Range("E2").value = Summary(1, UBound(Summary, 2))
    
    'debug
    '’PˆÊC“¾ó‹µ•\‚ğ•\¦
    'Debug.Print "Summary = "
    'For i = 1 To UBound(Summary)
    '    For c = 1 To UBound(Summary, 2)
    '        Debug.Print Summary(i, c),
    '    Next c
    '    Debug.Print
    'Next i
    'Debug.Print
    
   'Stop
    
    'o—Í—pƒuƒbƒNAƒV[ƒg‚Ì€”õ
    Dim WbOut As Workbook
    Dim WsOut As Worksheet
    
    outname = "Œ‹‰Ê_" & Format(Date, "YYYY-MM-DD")
    outpath = WsMain.Range("B5").value
    If Dir(outpath & "€" & outname & ".xlsx") <> "" Then
        Set WbOut = Workbooks.Open(outpath & "€" & outname & ".xlsx")
        WbOut.Worksheets.Add before:=WbOut.Worksheets(1)
    Else
        Set WbOut = Workbooks.Add
        WbOut.SaveAs Filename:=outpath & "€" & outname & ".xlsx"
    End If
    Set WsOut = WbOut.Worksheets(1)
    WsOut.Name = outname & Format(Time, " HHMMSS")   'ƒ[ƒNƒV[ƒg–¼Ad•¡Aã‘‚«‰ñ”ğ
        
    WsOut.Range("A1").value = "Šw¶”Ô†"
    WsOut.Range("B1").value = "ŠúŠÔ“à’PˆÊ”"
    For col = 2 To UBound(Summary, 2)
        WsOut.Cells(1, 3 + col - 2).value = Summary(1, col)  '2022-10  ƒXƒ^ƒCƒ‹i•¶š—ñj
        'WsOut.Cells(1, 3 + col - 2).NumberFormatLocal = "yy-mm-dd"
    Next col
    headingaddr = "A1:" & WsOut.Cells(1, UBound(Summary, 2) + 1).Address()
    'Debug.Print headingaddr
    WsOut.Range(headingaddr).BorderAround (xlContinuous)
    WsOut.Range(headingaddr).Interior.Color = RGB(224, 224, 224)   '–¾‚é‚¢ŠDF
    
    'stop
    
    Dim SRecords() As Variant
    Dim en4semes As Variant
    Dim lastSemester As String
    Dim lastDate As Date
    
    lastSemester = Summary(1, UBound(Summary, 2))
    Dim resArray() As String
    resArray = Split(lastSemester, "-")
    lastDate = DateSerial(resArray(0), resArray(1), 1)
    Debug.Print "lastSemesterStartDate = ", lastSemesterStartDate
    'Stop
    
    Dim ix As Integer
    Dim sid As String
    Dim fromSemes As String, toSemes As String
    Dim semeslist() As String
    
    ReDim semeslist(1 To UBound(Summary, 2) - 1)
    For col = 2 To UBound(Summary, 2)
        semeslist(col - 1) = Summary(1, col)
    Next col
    
    For ix = 2 To UBound(Summary)
    
        sid = Summary(ix, 1)
        SRecords = ExtractSIDRecords(Records, sid)
        
        Debug.Print "Output if  ExtractSIDRecords(Records, sid) = "
        For i = 1 To UBound(SRecords)
            Debug.Print SRecords(i, 1), SRecords(i, 2), SRecords(i, 3), SRecords(i, 4)
        Next i
        Debug.Print
        Debug.Print "semeslist(*) = "
        For i = 1 To UBound(semeslist)
            Debug.Print semeslist(i), VarType(semeslist(i))
        Next i
        Debug.Print

        'Summary ‚ÌŒ©o‚µs‚Ì‚İQÆ
        en4semes = EnrolledLast4Semesters(SRecords, semeslist)
        fromSemes = en4semes(LBound(en4semes))
        toSemes = en4semes(UBound(en4semes))
        '‚SŠwŠúˆÈ~‚ª‹xŠwŠúŠÔ‚Ìê‡‚É‚ÍWŒvI—¹ŠwŠú‚Ü‚ÅWŒvŠúŠÔ‚ÉŠÜ‚ß‚é
        If SRecords(UBound(SRecords), 3) = "‹xŠw" Then
            toSemes = lastSemester
        End If
        
        Debug.Print "en4semes for " & sid & ": "
        For i = 1 To UBound(en4semes)
            Debug.Print "'" & en4semes(i) & "'"
        Next i
        Debug.Print "involking totalschoolcredit w/ from ", fromSemes, " to ", toSemes
        sc = TotalScoolCredit(Summary, ix, fromSemes, toSemes)
        
        WsOut.Range("A" & (ix)).value = sid
        WsOut.Range("A" & (ix)).Interior.Color = RGB(239, 239, 239)
        WsOut.Range("B" & (ix)).value = sc
        If sc < 40 And SRecords(UBound(SRecords), 3) <> "‘ŞŠw" Then
            WsOut.Range("B" & (ix)).Interior.Color = RGB(255, 192, 128)
        End If
        
        For col = 2 To UBound(Summary, 2)
            WsOut.Cells(ix, 3 + col - 2).value = Summary(ix, col)
            sem = Summary(1, col)
            
            If fromSemes <= sem And sem <= toSemes Then
                WsOut.Cells(ix, 3 + col - 2).Interior.Color = RGB(255, 255, 128)
            End If
            flag = False
            For j = 1 To UBound(en4semes)
                If sem = en4semes(j) Then
                    flag = True
                    Exit For
                End If
            Next j
            If flag Then
                If sem = en4semes(LBound(en4semes)) Or sem = en4semes(UBound(en4semes)) Then
                    WsOut.Cells(ix, 3 + col - 2).Interior.Color = RGB(192, 192, 255)
                Else
                    WsOut.Cells(ix, 3 + col - 2).Interior.Color = RGB(224, 224, 255)
                End If
            End If
        Next col
        WsOut.Range("A" & (ix) & ":" & WsOut.Cells(ix, UBound(Summary, 2) + 1).Address()) _
        .BorderAround (xlContinuous)

    Next ix

    Erase en4semes
    
    WbOut.Save
    WbOut.Activate
    
    Erase Records
    Erase Summary
    
    Set WkMain = Nothing
    Set WkRecords = Nothing
    Set WkSummary = Nothing
End Sub

Public Function findRowInColumn(tbl As Variant, col As Integer, tgt As String) As Integer
    'search for tgt in column col
    For rx = LBound(tbl) To UBound(tbl)
        If tbl(rx, col) = tgt Then
           findRowInColumn = rx
           Exit Function
        End If
    Next rx
    findRowInColumn = LBound(tbl) - 1
End Function

Public Function TotalScoolCredit(summtbl As Variant, rix As Integer, fromSemes As String, toSemes As String) As Integer
    Dim bottomRow As Integer
    Dim sum As Integer
    
    Debug.Print "debug: TotalSchoolCredit", "fromSemes = ", fromSemes
    'Œ©o‚µs‚Ì“ú•t‚ğ’Tõ
    sum = 0
    For c = 2 To UBound(summtbl, 2)
        semes = summtbl(1, c)
        'Debug.Print fromSemes, semes, toSemes, fromSemses <= semes, semes <= toSemes
        If fromSemes <= semes And semes <= toSemes Then
            sum = sum + summtbl(rix, c)
        End If
    Next c
    'Debug.Print sum
    'Debug.Print
    TotalScoolCredit = sum
End Function

Public Function ExtractSIDRecords(recs As Variant, sid As String) As Variant
    Dim srecs() As Variant
    Dim cnt As Integer
    Dim lastEvt As String
    
    'count records related to sid
    cnt = 0
    bgix = 0: edix = 0
    For ix = 1 To UBound(recs)
        If recs(ix, 1) = sid Then
            cnt = cnt + 1
            If bgix = 0 Then bgix = ix
            edix = ix
        End If
    Next ix
    If cnt = 0 Then
        MsgBox "ExtractSIDRecords: Error; couldn't find records for sid = " & sid
        'Stop
        ExtractSIDRecords = srecs
        Exit Function
    End If
    
    'debug
    'Debug.Print "sid, bgx, edx, cnt", sid, bgix, edix, cnt
    
    ReDim srecs(1 To cnt, 4)
    cx = 1
    For ix = 1 To UBound(recs)
        If recs(ix, 1) = sid Then
            srecs(cx, 1) = recs(ix, 1)
            srecs(cx, 2) = recs(ix, 2)
            srecs(cx, 3) = recs(ix, 3)
            srecs(cx, 4) = recs(ix, 4)
            cx = cx + 1
        End If
    Next ix
    Debug.Print
    
    'debug
    'For i = 1 To UBound(srecs)
    '    Debug.Print i, srecs(i, 1), srecs(i, 2), srecs(i, 3), srecs(i, 4)
    'Next i
    
    ExtractSIDRecords = srecs
End Function

Public Function EnrolledLast4Semesters(srecs As Variant, semeslist As Variant)
    
    Dim Semesters() As KeyValuePair 'ŠwŠú–¼ string ‚ÆİŠwiTruej‹xŠwiFalsejvariant ‚Ì•Ê
    Dim sbpair As KeyValuePair      '‘€ì—p
    Dim result() As String 'o—Í
    
    Debug.Print
    Debug.Print "computing the enrolled last 4 semesters."
    
    ReDim Semesters(1 To UBound(semeslist))
    For i = 1 To UBound(semeslist)
        Semesters(i).key = Format(CDate(semeslist(i)), "yyyy-mm")
        Semesters(i).value = True
    Next i

    '‚ ‚éŠw¶”Ô†‚É‚Â‚¢‚Ä‚ÌƒCƒxƒ“ƒg—ğ‚©‚çİŠwŠwŠú‚Ì—ñ‚ğ¶¬
    For ix = 1 To UBound(srecs)
        'Debug.Print "srecs(" & ix & ") = (", srecs(ix, 1), ", ", srecs(ix, 2), ", ", srecs(ix, 3), ", ", srecs(ix, 4), ")"
        evtname = srecs(ix, 4)
        evtstart = Format(CDate(srecs(ix, 2)), "yyyy-mm")
        If srecs(ix, 3) <> "" Then
            evtend = Format(CDate(srecs(ix, 3)), "yyyy-mm")
        Else
            evtend = ""
        End If
        Debug.Print evtname, evtstart, evtend
        If evtname = "“üŠw" Then
            For jx = 1 To UBound(Semesters)
                If evtstart <= Semesters(jx).key Then
                    Semesters(jx).value = True
                End If
            Next jx
        End If
        If evtname = "‹xŠw" Then
            For jx = 1 To UBound(Semesters)
                If evtstart <= Semesters(jx).key Then
                    Semesters(jx).value = False
                End If
                If evtend < Semesters(jx).key Then
                    Exit For
                End If
            Next jx
        End If
    Next ix
    
    'ÅŒã‚©‚çİŠw‚µ‚½ 4 ŠwŠú‚Ì”²‚«o‚µiŒ‹‰Ê‚Í‹t‡j
    ReDim result(1 To UBound(Semesters))
    counter = 0
    For i = UBound(Semesters) To 1 Step -1
        Debug.Print i, Semesters(i).key, Semesters(i).value
        If Semesters(i).value = True Then
            counter = counter + 1
            result(counter) = Semesters(i).key
        End If
        If counter >= 4 Then Exit For
    Next i
    Debug.Print counter
    Debug.Print "result", counter, UBound(result)
    'Å‘å4ŠwŠú•ª‚¾‚¯”z—ñ‚ğc‚·i‚Ü‚¾‹t‡j
    ReDim Preserve result(1 To counter)
    '”z—ñ‚Ì“à—e‚Ì”½“]
    For i = 1 To counter / 2
        h = result(i)
        t = result(counter - i + 1)
        result(i) = t
        result(counter - i + 1) = h
    Next i
    EnrolledLast4Semesters = result
End Function

Sub collect()
    Dim ws As Worksheet
    Dim outws As Worksheet
    Dim rng As Range
    Dim rownum As Long
    Dim colnum As Long
    Dim criteria1 As String
    Dim criteria2 As String
    
    ' ‘ÎÛ‚ÌƒV[ƒg‚ğw’è
    Set ws = ThisWorkbook.Sheets("ŠwĞˆÚ“®")
    Set outws = ThisWorkbook.Sheets("ƒƒCƒ“")
    
    ' ƒf[ƒ^‚ÌÅIsAÅI—ñ
    rownum = ws.Cells(ws.Rows.Count, "B").End(xlUp).row
    colnum = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column
    
    ' ƒf[ƒ^‚Ì”ÍˆÍ‚ğw’è
     Set rng = ws.Range(ws.Cells(2, 1), ws.Cells(rownum, colnum))
    
    outws.Range("A1").value = "the number of data"
    outws.Range("b1").value = rownum
    
End Sub

Sub ğŒ’Šo()
    Dim dataSheet As Worksheet
    Dim resultSheet As Worksheet
    Dim dataRange As Range
    Dim LastRow As Long
    Dim sidRange As Range
    Dim dataRowNum As Long
    Dim temp
    
    'Application.ScreenUpdating = False

    ' ƒf[ƒ^‚ªŠi”[‚³‚ê‚Ä‚¢‚éƒV[ƒg‚ÆŒ‹‰Ê‚ğ•\¦‚·‚éƒV[ƒg‚ğw’è
    Set dataSheet = ThisWorkbook.Sheets("ŠwĞˆÚ“®")
    Set resultSheet = ThisWorkbook.Sheets("ƒƒCƒ“")
    
    dataRowNum = dataSheet.Cells(dataSheet.Rows.Count, "A").End(xlUp).row
    Set dataRange = dataSheet.Range("A1:C" & dataRowNum)
    Set sidRange = dataSheet.Range("A1:A" & dataRowNum)
    sidRange.AdvancedFilter Action:=xlFilterCopy, copytorange:=resultSheet.Range("F1:f" & dataRowNum), Unique:=True
    
    resultSheet.Range("A3:C" & dataRowNum).Clear
    
    ' ’ŠoğŒ‚ğw’è
    criteria1 = "221X1103"
    'criteria2 = "ğŒ2"

    ' ƒf[ƒ^‚ğ’Šo
    dataSheet.Activate
    With dataRange
        .Sort Key1:=Range("B1"), order1:=xlAscending, Header:=xlYes
        .AutoFilter Field:=1, criteria1:=criteria1
    'dataRange.AutoFilter Field:=2, criteria1:=criteria2
    End With

    ' ’Šoƒf[ƒ^‚ğŒ‹‰ÊƒV[ƒg‚ÉƒRƒs[
    dataSheet.UsedRange.SpecialCells(xlCellTypeVisible).Copy resultSheet.Range("A3")
    dataSheet.AutoFilterMode = False
    
    resultSheet.Activate
End Sub


Public Sub QuickSort(vArray As Variant, inLow As Long, inHi As Long)
  Dim pivot   As Variant
  Dim tmpSwap As Variant
  Dim tmpLow  As Long
  Dim tmpHi   As Long

  tmpLow = inLow
  tmpHi = inHi

  pivot = vArray((inLow + inHi)  2))

  While (tmpLow <= tmpHi)
     While (vArray(tmpLow) < pivot And tmpLow < inHi)
        tmpLow = tmpLow + 1
     Wend

     While (pivot < vArray(tmpHi) And tmpHi > inLow)
        tmpHi = tmpHi - 1
     Wend

     If (tmpLow <= tmpHi) Then
        tmpSwap = vArray(tmpLow)
        vArray(tmpLow) = vArray(tmpHi)
        vArray(tmpHi) = tmpSwap
        tmpLow = tmpLow + 1
        tmpHi = tmpHi - 1
     End If
  Wend

  If (inLow < tmpHi) Then QuickSort vArray, inLow, tmpHi
  If (tmpLow < inHi) Then QuickSort vArray, tmpLow, inHi
End Sub

