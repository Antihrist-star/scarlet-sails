# PowerShell команды для анализа структуры проекта
# Запустите эти команды на вашей машине

Write-Host "=== 1. МОНЕТЫ И ДАННЫЕ ===" -ForegroundColor Cyan
Write-Host "Команда: ls data/raw или dir data\raw"
Write-Host ""

Write-Host "=== 2. МОДЕЛИ ===" -ForegroundColor Cyan
Write-Host "Команда: ls models или dir models"
Write-Host ""

Write-Host "=== 3. СКРИПТЫ ===" -ForegroundColor Cyan
Write-Host "Команда: ls scripts/*.py | head -20"
Write-Host ""

Write-Host "=== 4. GIT BRANCHES ===" -ForegroundColor Cyan
Write-Host "Локальные: git branch -a -v"
Write-Host "Все на GitHub: git branch -r -v"
Write-Host ""

Write-Host "=== 5. ПРОВЕРИТЬ ОСНОВНУЮ ВЕТКУ ===" -ForegroundColor Cyan
Write-Host "git log main --oneline | head -5"
Write-Host ""

Write-Host "=== 6. КАКИЕ ФАЙЛЫ В MAIN ===" -ForegroundColor Cyan
Write-Host "git ls-tree -r main --name-only | grep -E '(data|models)'"
