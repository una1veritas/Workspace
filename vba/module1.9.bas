Attribute VB_Name = "Module2"
Function GetRangeValues(sheetName As String, rangeAddress As String) As Variant
    Dim ws As Worksheet
    Dim data As Variant

    ' Set the worksheet object
    Set ws = ThisWorkbook.Worksheets(sheetName)
    
    ' Get the values in the specified range
    data = ws.Range(rangeAddress).value
    
    ' Return the values as a variant array
    GetRangeValues = data
End Function

Function TransposeArray(inputArray As Variant) As Variant
    Dim outputArray As Variant
    Dim i As Long, j As Long
    Dim rowCount As Long, colCount As Long

    ' Get the dimensions of the input array
    rowCount = UBound(inputArray, 1)
    colCount = UBound(inputArray, 2)
    
    ' Initialize the output array with transposed dimensions
    ReDim outputArray(1 To colCount, 1 To rowCount)
    
    ' Loop through the input array and transpose the data
    For i = 1 To rowCount
        For j = 1 To colCount
            outputArray(j, i) = inputArray(i, j)
        Next j
    Next i
    
    ' Return the transposed array
    TransposeArray = outputArray
End Function

Function ExtractRowsByValue(arr As Variant, colIndex As Integer, value As Variant) As Variant
    Dim i As Long, j As Long
    Dim tempArray() As Variant
    Dim matchCount As Long

    ' Initialize the match count
    matchCount = 0

    ' Loop through the array to count matches
    For i = LBound(arr, 1) To UBound(arr, 1)
        If arr(i, colIndex) = value Then
            matchCount = matchCount + 1
        End If
    Next i

    ' If no match found, return an empty array
    If matchCount = 0 Then
        ExtractRowsByValue = Array()
        Exit Function
    End If

    ' Resize the temporary array to hold the matching rows
    ReDim tempArray(1 To matchCount, LBound(arr, 2) To UBound(arr, 2))

    ' Copy matching rows to the temporary array
    matchCount = 0
    For i = LBound(arr, 1) To UBound(arr, 1)
        If arr(i, colIndex) = value Then
            matchCount = matchCount + 1
            For j = LBound(arr, 2) To UBound(arr, 2)
                tempArray(matchCount, j) = arr(i, j)
            Next j
        End If
    Next i

    ' Return the extracted rows as a new array
    ExtractRowsByValue = tempArray
End Function


Function IsEmptyRow(ByVal table As Variant, ByVal rowIndex As Long) As Boolean
    Dim colCount As Long
    Dim j As Long
    
    ' Determine number of columns in the table
    colCount = UBound(table, 2)
    
    ' Check if all cells in the specified row are empty
    For j = 1 To colCount
        If Len(Trim(table(rowIndex, j))) > 0 Then
            IsEmptyRow = False
            Exit Function
        End If
    Next j
    
    ' If all cells in the row are empty, return True
    IsEmptyRow = True
End Function

Function IsEmptyColumn(ByVal table As Variant, ByVal colIndex As Long) As Boolean
    Dim colCount As Long
    Dim j As Long
    
    ' Determine number of columns in the table
    colCount = UBound(table, 2)
    
    ' Check if all cells in the specified row are empty
    For j = 1 To colCount
        If Len(Trim(table(rowIndex, j))) > 0 Then
            IsEmptyRow = False
            Exit Function
        End If
    Next j
    
    ' If all cells in the row are empty, return True
    IsEmptyRow = True
End Function

Function RemoveEmptyRows(ByVal table As Variant) As Variant
    Dim i As Long, j As Long, k As Long
    Dim rowCount As Long, colCount As Long
    Dim nonEmptyRows As Long
    
    ' Determine number of rows and columns in the table
    rowCount = UBound(table, 1)
    colCount = UBound(table, 2)
    
    ' Count non-empty rows
    nonEmptyRows = 0
    For i = 1 To rowCount
        If Not IsEmptyRow(table, i) Then
           nonEmptyRows = nonEmptyRows + 1
        End If
    Next i
    
    ' Initialize the result array with the count of non-empty rows
    Dim result() As Variant
    ReDim result(1 To nonEmptyRows, 1 To colCount)
    
    ' Copy non-empty rows to the result array
    k = 1
    For i = 1 To rowCount
        If Not IsEmptyRow(table, i) Then
            For j = 1 To colCount
                result(k, j) = table(i, j)
            Next j
            k = k + 1
        End If
    Next i
    
    ' Return the result array
    RemoveEmptyRows = result
End Function

Sub PrintTable(tbl As Variant)
    ' Loop through the returned array and print the values
    For i = LBound(tbl, 1) To UBound(tbl, 1)
        Debug.Print i; " ";
        For j = LBound(tbl, 2) To UBound(tbl, 2)
            Debug.Print tbl(i, j) & " ";
        Next j
        Debug.Print
    Next i
End Sub

Sub TestGetRangeValues()
    Dim values As Variant, transed As Variant
    Dim i As Long, j As Long

    ' Call the function with the worksheet name and range address
    values = GetRangeValues("TransTable", "I2:CR10 ")
    transed = TransposeArray(values)
    PrintTable transed
    Debug.Print
    
    ' Loop through the returned array and print the values
    
    values = GetRangeValues("TransTable", "A12:F72")
    values = RemoveEmptyRows(values)
    PrintTable values
End Sub
