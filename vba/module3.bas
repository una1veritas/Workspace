Attribute VB_Name = "Module3"
Sub ArrayToWorksheetCells(arr As Variant, startcell As Range)
    Dim i As Long, j As Long
    Dim itemArray As Variant
    itemArray = Array(5, 1, 2, 3, 4, 8, 6)

    ' Ensure the input is a 2D array
    If Not IsArray(arr) Then
        MsgBox "Input is not an array"
        Exit Sub
    End If

    ' Get the dimensions of the array
    Dim rowCount As Long, colCount As Long
    rowCount = UBound(arr, 1) - LBound(arr, 1) + 1
    colCount = UBound(arr, 2) - LBound(arr, 2) + 1

    ' Loop through the array and put values into the worksheet
    For i = LBound(arr, 1) To UBound(arr, 1)
        For j = LBound(itemArray, 1) To UBound(itemArray, 1)
            startcell.Cells(i - LBound(arr, 1) + 1, j - LBound(arr, 2) + 1).value = arr(i, itemArray(j))
        Next j
    Next i
End Sub
 
Sub MakeApplFrom()
    Dim startcell As Range
    'Dim crossTableRange As Range
    Dim crossTable As Variant
    Dim wsApplForm As Worksheet
    
    ' Define the starting cell
    Set wsApplForm = ThisWorkbook.Sheets("知能")
    Set startcell = wsApplForm.Range("B14")

    '認定科目の表
    values = GetRangeValues("TransTable", "J2:CR9 ")
    tgtCourses = TransposeArray(values)
    
    'Debug.Print "after trans"
    'PrintTable (tgtCourses)
    'Debug.Print
         
    '既修得科目（根拠科目）の表
    evdcourses = GetRangeValues("TransTable", "B11:G80")
    
    'Debug.Print "all evd courses"
    'PrintTable (evdcourses)
    'Debug.Print
    
    '既修得科目から認定申請科目への単位読み替え表
    crossTable = GetRangeValues("TransTable", "J11:CR80")
    rowCount = 0
    theFirstRow = False
    For tix = LBound(tgtCourses, 1) To UBound(tgtCourses, 1)
        If tgtCourses(tix, 8) <> "N" Then
            'Debug.Print "tgt"; tix; tgtCourses(tix, 5); tgtCourses(tix, 9)
            theFirstRow = True
            ' Target Couse name appear only at the first row.
            For eix = LBound(evdcourses, 1) To UBound(evdcourses, 1)
                part = val(Trim(crossTable(eix, tix)))
                If part > 0 Then
                    '申請科目または根拠科目のための行挿入
                    wsApplForm.Rows(startcell.Row + rowCount).Insert
                    rowCount = rowCount + 1
                    
                    '最初の行のみ認定科目名を記入（セル結合されてしまう可能性がある）
                    If theFirstRow Then
                        startcell.Cells(rowCount, 1).value = tgtCourses(tix, 5)
                        startcell.Cells(rowCount, 2).value = tgtCourses(tix, 1)
                        startcell.Cells(rowCount, 3).value = tgtCourses(tix, 2)
                        startcell.Cells(rowCount, 4).value = tgtCourses(tix, 3)
                        startcell.Cells(rowCount, 5).value = tgtCourses(tix, 4)
                        startcell.Cells(rowCount, 7).value = tgtCourses(tix, 6)
                        startcell.Cells(rowCount, 6).value = tgtCourses(tix, 7)
                        If tgtCourses(tix, 8) = "Y" Then
                            '対面実施科目として申請
                            startcell.Cells(rowCount, 8).value = "〇"
                        ElseIf tgtCourses(tix, 8) = "R" Then
                            '遠隔科目として申請（学科の履修課程表上の区別に関係なく、根拠科目の構成から対面か遠隔かの区分の希望を判定
                            startcell.Cells(rowCount, 9).value = "〇"
                        End If
                        theFirstRow = False
                    End If
                    If evdcourses(eix, 5) = "" Then
                        '根拠が対面科目
                        startcell.Cells(rowCount, 10).value = evdcourses(eix, 2)
                        startcell.Cells(rowCount, 11).value = evdcourses(eix, 6)
                        startcell.Cells(rowCount, 12).value = part
                        startcell.Cells(rowCount, 13).value = evdcourses(eix, 3)
                        startcell.Cells(rowCount, 14).value = evdcourses(eix, 4)
                        startcell.Cells(rowCount, 15).value = evdcourses(eix, 1)
                    Else
                        '根拠が遠隔科目
                        startcell.Cells(rowCount, 17).value = evdcourses(eix, 2)
                        startcell.Cells(rowCount, 18).value = evdcourses(eix, 6)
                        startcell.Cells(rowCount, 19).value = part
                        startcell.Cells(rowCount, 20).value = evdcourses(eix, 3)
                        startcell.Cells(rowCount, 21).value = evdcourses(eix, 4)
                        startcell.Cells(rowCount, 22).value = evdcourses(eix, 1)
                    End If
                End If
            Next eix
            End If
    Next tix
    'Debug.Print

    ' Call the procedure to put the array contents into the worksheet cells
    'ArrayToWorksheetCells tgtcourses, startcell
End Sub
