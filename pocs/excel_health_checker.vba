Function GetHttpStatusCode(url As String) As Integer
    Dim http As Object
    Set http = CreateObject("MSXML2.XMLHTTP")

    On Error Resume Next
    http.Open "GET", url, False
    http.Send
    GetHttpStatusCode = http.Status
    On Error GoTo 0
End Function
