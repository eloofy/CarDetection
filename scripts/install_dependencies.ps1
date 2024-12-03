# Function for error checking and exiting in case of failure
function Check-Error {
    param (
        [int]$exitCode,
        [string]$message
    )
    
    if ($exitCode -ne 0) {
        Write-Host "$message" -ForegroundColor Red
        Exit $exitCode
    }
}

# Check for Python in PATH
$pythonCheck = Get-Command python -ErrorAction SilentlyContinue

if (-not $pythonCheck) {
    Write-Host "Python not found. Installing Python..."
    Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.11.6/python-3.11.6-amd64.exe -OutFile python-installer.exe
    $installPythonResult = Start-Process .\python-installer.exe -ArgumentList '/quiet', 'InstallAllUsers=1', 'PrependPath=1' -Wait -PassThru
    Check-Error $installPythonResult.ExitCode "Error while installing Python"
    Remove-Item python-installer.exe
} else {
    Write-Host "Python is already installed"
}

# Check for pip
$pipCheck = Get-Command pip -ErrorAction SilentlyContinue
if (-not $pipCheck) {
    Write-Host "pip not found. Installing pip..."
    $installPipResult = python -m ensurepip
    Check-Error $LASTEXITCODE "Error while installing pip"
} else {
    Write-Host "pip is already installed"
}

# Check for Poetry
$poetryCheck = Get-Command poetry -ErrorAction SilentlyContinue
if (-not $poetryCheck) {
    Write-Host "Poetry not found. Installing Poetry..."
    $installPoetryResult = Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing | python -
    Check-Error $LASTEXITCODE "Error while installing Poetry"
} else {
    Write-Host "Poetry is already installed"
}

# Installing dependencies using Poetry
Write-Host "Installing dependencies using Poetry..."
$installDependenciesResult = poetry install
Check-Error $LASTEXITCODE "Error while installing dependencies with Poetry"
