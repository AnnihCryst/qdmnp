[CmdletBinding()]
param(
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'
$RecommendedUvVersion = [version]'0.12.5'

Write-Host ''
Write-Host '=== QD-MNP: настройка окружения ===' -ForegroundColor Cyan
Write-Host ''

$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if ($null -eq $uvCommand) {
    Write-Host '[ОШИБКА] uv не найден в PATH.' -ForegroundColor Red
    Write-Host ''
    Write-Host 'Этот проект использует uv для управления зависимостями.'
    Write-Host 'Установите его одним из способов и запустите скрипт снова:'
    Write-Host ''
    Write-Host '  winget install --id astral-sh.uv -e'
    Write-Host ''
    Write-Host 'или:'
    Write-Host ''
    Write-Host '  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    Write-Host ''
    Write-Host 'Инструкция: https://docs.astral.sh/uv/getting-started/installation/'
    exit 1
}

Push-Location -LiteralPath $PSScriptRoot
try {
    $uvVersionOutput = (& uv --version) | Out-String
    if ($uvVersionOutput -match 'uv\s+v?([0-9]+(?:\.[0-9]+)+)') {
        $detectedUvVersion = [version]$Matches[1]
        Write-Host ("[OK] uv найден: v{0} ({1})" -f $detectedUvVersion, $uvCommand.Source)
        if ($detectedUvVersion -lt $RecommendedUvVersion) {
            Write-Host ("[ВНИМАНИЕ] Рекомендуемая версия uv - {0} или новее. Обновление: uv self update" -f $RecommendedUvVersion) -ForegroundColor Yellow
        }
    }
    else {
        Write-Host '[ВНИМАНИЕ] Не удалось определить версию uv.' -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host '[1/3] Создание окружения и установка зависимостей по uv.lock...'
    & uv sync
    if ($LASTEXITCODE -ne 0) {
        Write-Host '[ОШИБКА] uv sync завершился с ошибкой.' -ForegroundColor Red
        exit 1
    }

    if (-not $SkipVerify) {
        Write-Host ''
        Write-Host '[2/3] Проверка импортов...'
        & uv run --no-sync python -c "import sys, numpy, scipy, matplotlib; print('python     ', sys.version.split()[0]); print('numpy      ', numpy.__version__); print('scipy      ', scipy.__version__); print('matplotlib ', matplotlib.__version__)"
        if ($LASTEXITCODE -ne 0) {
            Write-Host '[ОШИБКА] Проверка импортов не прошла.' -ForegroundColor Red
            exit 1
        }
    }
    else {
        Write-Host ''
        Write-Host '[2/3] Проверка импортов пропущена (-SkipVerify).'
    }

    Write-Host ''
    Write-Host '[3/3] Окружение готово.' -ForegroundColor Green
    Write-Host ''
    Write-Host 'Запуск скриптов проекта (примеры из README):'
    Write-Host ''
    Write-Host '  .\.venv\Scripts\python.exe qd_mnp_linear_spectrum.py --energy-min-ev 2.0 --energy-max-ev 2.08 --target-ev 2.042'
    Write-Host '  .\.venv\Scripts\python.exe qd_mnp_fano_scan.py --top 10'
    Write-Host ''
    Write-Host 'Либо короче, через uv:'
    Write-Host ''
    Write-Host '  uv run qd_mnp_linear_spectrum.py --energy-min-ev 2.0 --energy-max-ev 2.08 --target-ev 2.042'
    Write-Host ''
}
finally {
    Pop-Location
}
