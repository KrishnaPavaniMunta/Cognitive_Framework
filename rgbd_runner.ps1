#!/usr/bin/env powershell
<#
.SYNOPSIS
    Convenience runner for RGB-D spatial twin development scripts
.DESCRIPTION
    Runs rgbd_spatial_twin.py from the organized hospital_detector_longterm/rgbd_development folder
.PARAMETER Command
    Command to run: view, export, view-db, etc.
.PARAMETER Sequence
    Which sequence: freiburg1 (default) or freiburg3
.PARAMETER Frames
    Max frames to process (optional)
.PARAMETER OutputVideo
    Video output path (optional, for export command)
#>

param(
    [Parameter(Position=0)][string]$Command = "view",
    [Parameter(Position=1)][string]$Sequence = "freiburg1",
    [int]$Frames = 0,
    [string]$OutputVideo = ""
)

$ScriptsDir = "hospital_detector_longterm\rgbd_development\scripts"
$DataDir = "hospital_detector_longterm\rgbd_development\data"
$OutputDir = "hospital_detector_longterm\rgbd_development\output"
$ExportDir = "$OutputDir\exports"
$DetectionDir = "$OutputDir\detections"
$LogDir = "$OutputDir\logs"

# Map sequence names to actual paths
$SequencePaths = @{
    "freiburg1" = "$DataDir\rgbd_dataset_freiburg1_xyz"
    "freiburg3" = "$DataDir\rgbd_dataset_freiburg3_long_office_household"
}

if (-not $SequencePaths.ContainsKey($Sequence)) {
    Write-Host "Error: Sequence '$Sequence' not found. Valid options: freiburg1, freiburg3" -ForegroundColor Red
    exit 1
}

$SequencePath = $SequencePaths[$Sequence]
$DbPath = "$OutputDir\hospital_twin.db"
$PythonScript = "$ScriptsDir\rgbd_spatial_twin.py"
$Python = "C:/Users/Krishna.Munta/AppData/Local/Python/pythoncore-3.14-64/python.exe"

New-Item -ItemType Directory -Path $ExportDir, $DetectionDir, $LogDir -Force | Out-Null

# Build command based on operation
$Args = @("$PythonScript", "--sequence-root", $SequencePath)

switch ($Command) {
    "view" {
        $Args += "--no-db", "--wait-ms", "15"
        Write-Host "Viewing $Sequence RGB-D stream..." -ForegroundColor Cyan
    }
    "export" {
        if ([string]::IsNullOrEmpty($OutputVideo)) {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $OutputVideo = "$ExportDir\rgbd_${Sequence}_${timestamp}.mp4"
        }
        $Args += "--no-db", "--output-video", $OutputVideo
        Write-Host "Exporting $Sequence to $OutputVideo..." -ForegroundColor Cyan
    }
    "export-db" {
        if ([string]::IsNullOrEmpty($OutputVideo)) {
            $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
            $OutputVideo = "$ExportDir\rgbd_${Sequence}_${timestamp}.mp4"
        }
        $Args += "--db", $DbPath, "--output-video", $OutputVideo
        Write-Host "Exporting $Sequence with spatial memory logging to $OutputVideo..." -ForegroundColor Cyan
    }
    "view-db" {
        $Args += "--db", $DbPath, "--wait-ms", "15"
        Write-Host "Viewing $Sequence with spatial memory logging..." -ForegroundColor Cyan
    }
    default {
        Write-Host "Unknown command: $Command" -ForegroundColor Red
        Write-Host ""
        Write-Host "Available commands:" -ForegroundColor Yellow
        Write-Host "  view       - View RGB-D stream (no database)" -ForegroundColor White
        Write-Host "  export     - Export RGB-D to MP4 (no database)" -ForegroundColor White
        Write-Host "  view-db    - View RGB-D with spatial memory logging" -ForegroundColor White
        Write-Host "  export-db  - Export RGB-D to MP4 with spatial memory logging" -ForegroundColor White
        Write-Host ""
        Write-Host "Usage: .\rgbd_runner.ps1 <command> [sequence] [options]" -ForegroundColor Yellow
        Write-Host "  sequence: freiburg1 (default) or freiburg3" -ForegroundColor Gray
        exit 1
    }
}

# Add max frames if specified
if ($Frames -gt 0) {
    $Args += "--max-frames", $Frames
}

# Execute
& $Python @Args
