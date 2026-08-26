' Novel Studio launcher: double-click = start local server (hidden)
' + open UI in an app-mode window (Edge/Chrome --app, no address bar).
Option Explicit
Const PORT = 8600

Dim Wsh, FSO, base
Set Wsh = CreateObject("WScript.Shell")
Set FSO = CreateObject("Scripting.FileSystemObject")
base = FSO.GetParentFolderName(WScript.ScriptFullName)

' ---------- locate pythonw.exe (no console window) ----------
Dim pyCands(3), py, c
pyCands(0) = "C:\Python314\pythonw.exe"
pyCands(1) = Wsh.ExpandEnvironmentStrings("%LocalAppData%") & "\Programs\Python\Python314\pythonw.exe"
pyCands(2) = Wsh.ExpandEnvironmentStrings("%LocalAppData%") & "\Programs\Python\Python313\pythonw.exe"
pyCands(3) = "pythonw.exe"
py = ""
For Each c In pyCands
    If c = "pythonw.exe" Or FSO.FileExists(c) Then
        py = c
        Exit For
    End If
Next
If py = "" Then
    MsgBox "Python 3.10+ not found. Please install Python first.", 16, "Novel Studio"
    WScript.Quit
End If

' ---------- start server (a second instance exits silently if port busy) ----------
Wsh.Run """" & py & """ """ & base & "\novel_studio.py"" --port " & PORT, 0, False

' ---------- wait until server is ready (up to ~15s) ----------
Dim http, i, up
up = False
For i = 0 To 30
    On Error Resume Next
    Set http = CreateObject("MSXML2.ServerXMLHTTP")
    http.open "GET", "http://127.0.0.1:" & PORT & "/api/state", False
    http.send
    If Err.Number = 0 Then
        If http.status = 200 Then up = True
    End If
    On Error GoTo 0
    If up Then Exit For
    WScript.Sleep 500
Next
If Not up Then
    MsgBox "Server start timeout. Run novel_studio.py manually to see the error.", _
           16, "Novel Studio"
    WScript.Quit
End If

' ---------- open UI as a standalone app window ----------
Dim apps(3), exe
apps(0) = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
apps(1) = "C:\Program Files\Microsoft\Edge\Application\msedge.exe"
apps(2) = "C:\Program Files\Google\Chrome\Application\chrome.exe"
apps(3) = "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
exe = ""
For Each c In apps
    If FSO.FileExists(c) Then
        exe = c
        Exit For
    End If
Next
If exe <> "" Then
    Wsh.Run """" & exe & """ --app=http://127.0.0.1:" & PORT & "/", 1, False
Else
    Wsh.Run "cmd.exe /c start """" http://127.0.0.1:" & PORT & "/", 0, False
End If
