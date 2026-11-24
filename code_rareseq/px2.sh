#!/bin/bash
# px2.sh — 在 px2 服务器上执行命令（密钥认证 + ControlMaster 连接复用）
# 用法: ./px2.sh '远程命令'   或   ./px2.sh < script.sh
exec ssh -p 29375 \
  -o ControlMaster=auto \
  -o ControlPath="$HOME/.ssh/cm/px2-%r@%h:%p" \
  -o ControlPersist=3600 \
  -o ServerAliveInterval=20 \
  -o ServerAliveCountMax=6 \
  -o ConnectTimeout=30 \
  -o StrictHostKeyChecking=accept-new \
  -i "$HOME/.ssh/id_ed25519" \
  root@px2-jcy.matpool.com "$@"