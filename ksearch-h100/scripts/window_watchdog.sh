#!/usr/bin/env bash
# 窗口看门狗：每 5 分钟检查卡池锁，持锁超过 30 分钟的任务判为"评测挂死"，
# 杀掉其进程树（campaign 会收尸并重排该任务）。产出日志 /tmp/ksearch_watchdog.log
WINDOW_TIMEOUT_SEC=1800
while true; do
  for lock in /tmp/ksearch_gpu_pool/gpu*.lock; do
    [ -f "$lock" ] || continue
    # 锁内容形如 pid=123 gpu=0 task=xxx ts=0920 19:26:38
    line=$(head -1 "$lock" 2>/dev/null)
    case "$line" in
      pid=*) ;;
      *) continue ;;
    esac
    pid=$(echo "$line" | grep -oE 'pid=[0-9]+' | cut -d= -f2)
    ts=$(echo "$line" | grep -oE 'ts=[0-9]+ [0-9:]+' | sed 's/ts=//')
    [ -n "$pid" ] && [ -n "$ts" ] || continue
    # 锁是否仍被持有（内容写了 pid 但 flock 可能已释放——看 flock 才算数）
    held=$(/home/ziming/miniconda3/envs/ksearch/bin/python3 -c "
import fcntl,sys
fh=open('$lock','a+')
try:
    fcntl.flock(fh.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(fh.fileno(), fcntl.LOCK_UN); print('no')
except OSError: print('yes')
finally: fh.close()")
    [ "$held" = "yes" ] || continue
    age=$(( $(date +%s) - $(date -d "${ts/\#/}" +%s 2>/dev/null || echo 0) ))
    if [ "$age" -gt "$WINDOW_TIMEOUT_SEC" ] && kill -0 "$pid" 2>/dev/null; then
      task=$(echo "$line" | grep -oE 'task=[^ ]+' | cut -d= -f2)
      echo "$(date '+%m-%d %H:%M') KILL hung window: $lock pid=$pid task=$task age=${age}s" >> /tmp/ksearch_watchdog.log
      pkill -9 -P "$pid" 2>/dev/null   # 先杀子进程（isolated runner）
      kill -9 "$pid" 2>/dev/null
    fi
  done
  sleep 300
done
