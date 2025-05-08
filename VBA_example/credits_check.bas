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
        fname = Application.GetOpenFilename("xlsx�t�@�C��(*.xlsx),*xlsx", 1, msg)
    End If
    GetFilename = fname
End Function

Public Function LastRow(wks As Variant, col As Integer) As Integer
    LastRow = wks.Cells(wks.Rows.Count, col).End(xlUp).row    'col ��ڂōŏI�s��T��
End Function

Public Function RightmostColumn(wks As Worksheet, r As Integer) As Integer
    RightmostColumn = wks.Cells(r, wks.Columns.Count).End(xlToLeft).Column    'row �s�ڂōŏI���T��
End Function

Sub �t�@�C���ݒ�()
    Dim wkst As Worksheet
    Set wkst = ThisWorkbook.Sheets("�R���g���[���p�l��")
    txt = GetFilename(wkst.Range("B2"), wkst.Range("A2").value & "���J��")
    wkst.Range("B2").value = txt
    txt = GetFilename(wkst.Range("B4"), wkst.Range("A4").value & "���J��")
    wkst.Range("B4").value = txt
    wkst.Range("B5").value = ThisWorkbook.Path
End Sub


Sub ���ѕs�U�`�F�b�N()
    Dim WsMain As Worksheet
    Dim WsRecords As Worksheet, WsSummary As Worksheet
    
    Dim Records As Variant
    Dim Summay As Variant
    
    Dim rangeTopLeft As Range  ' �\�̍�����Z�����w���͈�
    
    Set WsMain = ThisWorkbook.Sheets("�R���g���[���p�l��")
    
    Call �t�@�C���ݒ�
    
    '�P�ʏC���󋵃��[�N�V�[�g
    Set WsSummary = Workbooks.Open(WsMain.Range("B2").value).Worksheets(1)
    
    '�w�Јړ��L�^���[�N�V�[�g
    Set WsRecords = Workbooks.Open(WsMain.Range("B4").value).Worksheets(1)
    
    '�R���g���[���p�l�����[�N�V�[�g�̏��Ɋ�Â��w�Јٓ��C�x���g�̎��W
    Dim RColSID As String  '�w���ԍ����o����
    Dim RColBDate As String  '�ٓ��J�n���o����
    Dim RColEDate As String  '�ٓ��I�����o����
    Dim RColTrans As String  '�ٓ����e���o����
    Dim RecordsCount As Integer
                                            
    Set rangeTopLeft = WsRecords.Range(WsMain.Range("C4").value)  '�񌩏o�����[�Z����͈͂�
    Debug.Print "rangeTopLeft is ", VarType(rangeTopLeft), rangeTopLeft
    
    bottomRow = LastRow(WsRecords, rangeTopLeft.Column)  '1��ڂōs�����J�E���g
    RecordsCount = bottomRow - rangeTopLeft.row          '���C�x���g���i�f�[�^�s���j
    'Debug.Print rangeTopLeft.Address, rangeTopLeft.Column, rangeTopLeft.Row, bottomRow, RecordsCount

    For cx = 1 To RightmostColumn(WsRecords, rangeTopLeft.row)
        If rangeTopLeft.Offset(0, cx).value = "�w���ԍ�" Then
            RColSID = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("D4").value = RColSID
        ElseIf rangeTopLeft.Offset(0, cx).value = "�ٓ��J�n" Then
            RColBDate = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("E4").value = RColBDate
        ElseIf rangeTopLeft.Offset(0, cx) = "�ٓ��I��" Then
            RColEDate = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("F4").value = RColEDate
        ElseIf rangeTopLeft.Offset(0, cx) = "�ٓ����e" Then
            RColTrans = rangeTopLeft.Offset(0, cx).Address
            WsMain.Range("G4").value = RColTrans
        End If
    Next cx

    ' �w�Јړ��C�x���g���i�w���ԍ��A�ړ������������t�j�̏��Ń\�[�g
    WsRecords.Range(rangeTopLeft.Address).Resize(RecordsCount + 1, 15) _
        .Sort Key1:=WsRecords.Range(RColSID), Key2:=WsRecords.Range(RColBDate), Header:=xlYes
    '�z�� Records �Ɋi�[�i�f�[�^���̂݁j
    ReDim Records(RecordsCount + 1, 4)
    '�O�̂��ߓ��t�� vbDate �ɑ�����
    With WsRecords
        For rx = 1 To RecordsCount
            '�x�w�ȊO���Ƃ肱��
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

    '�P�ʏC���󋵂��w���ԍ����Ń\�[�g
    Set rangeTopLeft = WsSummary.Range(WsMain.Range("C2").value)  '�w���ԍ��A�P�ʂȂǂ̌��o���s�ō��[
    bottomRow = LastRow(WsSummary, rangeTopLeft.Column)  '1��ڂōs�����J�E���g
    rightColumn = RightmostColumn(WsSummary, rangeTopLeft.row)
    
    SIDsCount = bottomRow - rangeTopLeft.row               '�w���ԍ���
    semesCount = rightColumn - rangeTopLeft.Column
    
    WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, semesCount + 1).Sort Key1:=rangeTopLeft, Header:=xlYes
    Summary = WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, semesCount + 1).value
    'WsSummary.Range(rangeTopLeft.Address).Resize(SIDsCount + 1, SemesCount + 1).Select
    'Debug.Print SIDsCount, SemesCount
    
    '�P�ʏC���󋵕\�̌��o����i�N�x�w���j���擾
    With WsSummary.Range(rangeTopLeft.Address).Offset(-2, 1).Resize(2, UBound(Summary, 2) - 1)
        For cx = 1 To .Columns.Count
            '2022-10 �`��
            s = .Cells(1, cx).MergeArea(1, 1).value
            If .Cells(2, cx).value = "�O��" Then
                Summary(1, 1 + cx) = s & "-04"
            ElseIf .Cells(2, cx).value = "���" Then
                Summary(1, 1 + cx) = s & "-10"
            Else
                Summary(1, 1 + cx) = s & "-�G���[�w��"
            End If
        Next cx
    End With
    WsMain.Range("D2").value = Summary(1, 2)
    WsMain.Range("E2").value = Summary(1, UBound(Summary, 2))
    
    'debug
    '�P�ʏC���󋵕\��\��
    'Debug.Print "Summary = "
    'For i = 1 To UBound(Summary)
    '    For c = 1 To UBound(Summary, 2)
    '        Debug.Print Summary(i, c),
    '    Next c
    '    Debug.Print
    'Next i
    'Debug.Print
    
   'Stop
    
    '�o�͗p�u�b�N�A�V�[�g�̏���
    Dim WbOut As Workbook
    Dim WsOut As Worksheet
    
    outname = "����_" & Format(Date, "YYYY-MM-DD")
    outpath = WsMain.Range("B5").value
    If Dir(outpath & "�" & outname & ".xlsx") <> "" Then
        Set WbOut = Workbooks.Open(outpath & "�" & outname & ".xlsx")
        WbOut.Worksheets.Add before:=WbOut.Worksheets(1)
    Else
        Set WbOut = Workbooks.Add
        WbOut.SaveAs Filename:=outpath & "�" & outname & ".xlsx"
    End If
    Set WsOut = WbOut.Worksheets(1)
    WsOut.Name = outname & Format(Time, " HHMMSS")   '���[�N�V�[�g���A�d���A�㏑�����
        
    WsOut.Range("A1").value = "�w���ԍ�"
    WsOut.Range("B1").value = "���ԓ��P�ʐ�"
    For col = 2 To UBound(Summary, 2)
        WsOut.Cells(1, 3 + col - 2).value = Summary(1, col)  '2022-10  �X�^�C���i������j
        'WsOut.Cells(1, 3 + col - 2).NumberFormatLocal = "yy-mm-dd"
    Next col
    headingaddr = "A1:" & WsOut.Cells(1, UBound(Summary, 2) + 1).Address()
    'Debug.Print headingaddr
    WsOut.Range(headingaddr).BorderAround (xlContinuous)
    WsOut.Range(headingaddr).Interior.Color = RGB(224, 224, 224)   '���邢�D�F
    
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

        'Summary �̌��o���s�̂ݎQ��
        en4semes = EnrolledLast4Semesters(SRecords, semeslist)
        fromSemes = en4semes(LBound(en4semes))
        toSemes = en4semes(UBound(en4semes))
        '�S�w���ȍ~���x�w���Ԃ̏ꍇ�ɂ͏W�v�I���w���܂ŏW�v���ԂɊ܂߂�
        If SRecords(UBound(SRecords), 3) = "�x�w" Then
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
        If sc < 40 And SRecords(UBound(SRecords), 3) <> "�ފw" Then
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
    '���o���s�̓��t��T��
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
    
    Dim Semesters() As KeyValuePair '�w���� string �ƍ݊w�iTrue�j�x�w�iFalse�jvariant �̕�
    Dim sbpair As KeyValuePair      '����p
    Dim result() As String '�o��
    
    Debug.Print
    Debug.Print "computing the enrolled last 4 semesters."
    
    ReDim Semesters(1 To UBound(semeslist))
    For i = 1 To UBound(semeslist)
        Semesters(i).key = Format(CDate(semeslist(i)), "yyyy-mm")
        Semesters(i).value = True
    Next i

    '����w���ԍ��ɂ��ẴC�x���g������݊w�w���̗�𐶐�
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
        If evtname = "���w" Then
            For jx = 1 To UBound(Semesters)
                If evtstart <= Semesters(jx).key Then
                    Semesters(jx).value = True
                End If
            Next jx
        End If
        If evtname = "�x�w" Then
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
    
    '�Ōォ��݊w���� 4 �w���̔����o���i���ʂ͋t���j
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
    '�ő�4�w���������z����c���i�܂��t���j
    ReDim Preserve result(1 To counter)
    '�z��̓��e�̔��]
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
    
    ' �Ώۂ̃V�[�g���w��
    Set ws = ThisWorkbook.Sheets("�w�Јړ�")
    Set outws = ThisWorkbook.Sheets("���C��")
    
    ' �f�[�^�̍ŏI�s�A�ŏI��
    rownum = ws.Cells(ws.Rows.Count, "B").End(xlUp).row
    colnum = ws.Cells(1, ws.Columns.Count).End(xlToLeft).Column
    
    ' �f�[�^�͈̔͂��w��
     Set rng = ws.Range(ws.Cells(2, 1), ws.Cells(rownum, colnum))
    
    outws.Range("A1").value = "the number of data"
    outws.Range("b1").value = rownum
    
End Sub

Sub �������o()
    Dim dataSheet As Worksheet
    Dim resultSheet As Worksheet
    Dim dataRange As Range
    Dim LastRow As Long
    Dim sidRange As Range
    Dim dataRowNum As Long
    Dim temp
    
    'Application.ScreenUpdating = False

    ' �f�[�^���i�[����Ă���V�[�g�ƌ��ʂ�\������V�[�g���w��
    Set dataSheet = ThisWorkbook.Sheets("�w�Јړ�")
    Set resultSheet = ThisWorkbook.Sheets("���C��")
    
    dataRowNum = dataSheet.Cells(dataSheet.Rows.Count, "A").End(xlUp).row
    Set dataRange = dataSheet.Range("A1:C" & dataRowNum)
    Set sidRange = dataSheet.Range("A1:A" & dataRowNum)
    sidRange.AdvancedFilter Action:=xlFilterCopy, copytorange:=resultSheet.Range("F1:f" & dataRowNum), Unique:=True
    
    resultSheet.Range("A3:C" & dataRowNum).Clear
    
    ' ���o�������w��
    criteria1 = "221X1103"
    'criteria2 = "����2"

    ' �f�[�^�𒊏o
    dataSheet.Activate
    With dataRange
        .Sort Key1:=Range("B1"), order1:=xlAscending, Header:=xlYes
        .AutoFilter Field:=1, criteria1:=criteria1
    'dataRange.AutoFilter Field:=2, criteria1:=criteria2
    End With

    ' ���o�f�[�^�����ʃV�[�g�ɃR�s�[
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

