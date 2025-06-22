#!/usr/bin/env bash
# train.sh
# 실행:  chmod +x train.sh && ./train.sh

set -euo pipefail

# 학습 실행
python train.py

# 로그 파일 설정 (가장 최근 로그 파일 자동 찾기)
LOG_DIR="log/train"
if [ ! -d "$LOG_DIR" ]; then
    echo "❌ Log directory not found: $LOG_DIR"
    exit 1
fi

# 가장 최근 로그 파일 찾기
LOG_FILE=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
if [ -z "$LOG_FILE" ]; then
    echo "❌ No log files found in $LOG_DIR"
    exit 1
fi

echo "📋 Analyzing log file: $LOG_FILE"

# 학습 요약 로그 생성
SUMMARY_FILE="log_summary.txt"
grep -A 2 "completed" "${LOG_FILE}" > "${SUMMARY_FILE}"

# 에폭/종합 통계 출력
python - << 'PY'
import re, numpy as np, pathlib, textwrap

log_text = pathlib.Path("log_summary.txt").read_text()

# Fold별 best(최소) RMSE 찾기
best_per_fold = {}
for block in re.findall(r"(Fold\s+\d+.*?Val SCC:\s*[\d.]+)", log_text, re.DOTALL):
    header_line = block.splitlines()[0]
    
    fold_match = re.search(r"Fold\s+(\d+)", header_line)
    epoch_match = re.search(r"Epoch\s+\[(\d+)/", header_line)
    rmse_match = re.search(r"Val RMSE:\s*([\d.]+)", block)  # 전체 블록에서 검색
    pcc_match = re.search(r"Val PCC:\s*([\d.]+)", block)    # 전체 블록에서 검색
    scc_match = re.search(r"Val SCC:\s*([\d.]+)", block)    # 전체 블록에서 검색
    
    # 모든 매칭이 성공했을 때
    if all([fold_match, epoch_match, rmse_match, pcc_match, scc_match]):
        fold = int(fold_match.group(1))
        epoch = int(epoch_match.group(1))
        rmse = float(rmse_match.group(1))
        pcc = float(pcc_match.group(1))
        scc = float(scc_match.group(1))

        if fold not in best_per_fold or rmse < best_per_fold[fold]['rmse']:
            best_per_fold[fold] = dict(epoch=epoch, rmse=rmse, pcc=pcc, scc=scc)
    else:
        # 매칭 실패 시 디버깅 정보 출력
        print(f"⚠️ Skipping malformed log entry: {header_line[:100]}...")

# 출력
if not best_per_fold:
    print("❌ No valid fold results found in log file")
    print("📋 Log file content preview:")
    print(log_text[:500] + "..." if len(log_text) > 500 else log_text)
    exit(1)

print("📋 Best Epoch per Fold:")
val_r, val_p, val_s = [], [], []
for fold in sorted(best_per_fold):
    b = best_per_fold[fold]
    print(f"Fold {fold}: Epoch {b['epoch']} | Val RMSE: {b['rmse']:.4f}, "
          f"PCC: {b['pcc']:.4f}, SCC: {b['scc']:.4f}")
    val_r.append(b['rmse']); val_p.append(b['pcc']); val_s.append(b['scc'])

if val_r:  # 결과가 있을 때만 출력
    mean_std = lambda x: f"{np.mean(x):.4f} ± {np.std(x):.4f}"
    print(textwrap.dedent(f"""
        \n📊 Overall Performance (Best Val RMSE Epoch per Fold):
        Val RMSE: {mean_std(val_r)}
        Val PCC:  {mean_std(val_p)}
        Val SCC:  {mean_std(val_s)}
    """).strip())
else:
    print("❌ No valid results found in log file")
PY
