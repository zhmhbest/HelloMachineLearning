@SET IDEA_PATH=D:\ProgramFiles\Programmer\IDE\Jetbrains\ideaIU-2019.2.4.win\bin\idea64.exe
@CD /D %~dp0

:: 获取
@FOR /F "usebackq" %%i IN (`WHERE "python.exe" 2^>NUL`) DO @(
    @SET PYTHON=%%~dpnxi
)

:: 验证
@FOR /F "usebackq" %%i IN (`%PYTHON% -V`) DO @(
    @SET RETURN=%%i
)
@IF "Python" NEQ "%RETURN%" @(
    ECHO Press `Win` input `app exec`.
    PAUSE>NUL
)

:: 启动
@"%PYTHON%" "%CD%\com\zhmh\tf\environment.py" "%IDEA_PATH%"