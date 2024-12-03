.PHONY: all check-python check-pip check-poetry install-poetry install-python install


all: check-python check-pip check-poetry install


check-python:
	@if not exist "%SystemDrive%\Python*" ( \
		echo Python не найден. Устанавливаю Python... & \
		powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe -OutFile python-installer.exe; Start-Process .\python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -Wait; Remove-Item python-installer.exe" \
	) else ( \
		echo Python уже установлен \
	)

check-pip:
	@if not exist "%SystemDrive%\Python*\Scripts\pip.exe" ( \
		echo pip не найден. Устанавливаю pip... & \
		python -m ensurepip \
	) else ( \
		echo pip уже установлен \
	)

check-poetry:
	@if not exist "%USERPROFILE%\\.poetry\\bin\\poetry.exe" ( \
		echo Poetry не найден. Устанавливаю Poetry... & \
		powershell -Command "Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -" \
	) else ( \
		echo Poetry уже установлен \
	)


install:
	poetry install


install-python:
	@echo Устанавливаю Python...
	powershell -Command "Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe -OutFile python-installer.exe; Start-Process .\python-installer.exe -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1' -Wait; Remove-Item python-installer.exe"

install-poetry:
	@if not exist "%USERPROFILE%\\.poetry\\bin\\poetry.exe" ( \
		echo Устанавливаю Poetry... & \
		powershell -Command "Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -" \
	) else ( \
		echo Poetry уже установлен \
	)