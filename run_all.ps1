# Full evaluation pipeline for llm-hallucination-eval
# Run from project root. Ensure .env is configured with OPENAI_API_KEY.

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Step 1: Run evaluation (all 3 conditions)..." -ForegroundColor Cyan
python src/run_eval.py --all-conditions

Write-Host "`nStep 2: Score generations..." -ForegroundColor Cyan
python src/score.py

Write-Host "`nStep 3: Analyze and generate plots..." -ForegroundColor Cyan
python src/analyze.py

Write-Host "`nDone. Check results/ for scored.csv, summary.csv, and plots/" -ForegroundColor Green
