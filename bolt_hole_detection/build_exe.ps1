$ErrorActionPreference = "Stop"

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectDir

py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller .\RailwayBoltDetection.spec --noconfirm

Write-Host ""
Write-Host "Built: $ProjectDir\dist\RailwayBoltDetection\RailwayBoltDetection.exe"
