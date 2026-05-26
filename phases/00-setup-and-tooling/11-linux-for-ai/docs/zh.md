# Linux for AI

> 大多数人工智能都在 Linux 上运行。你需要掌握足够的知识，才不会卡壳。

**类型：** 学习
**语言：** --
**前置条件：** 阶段 0，第 01 课
**时间：** 约 30 分钟

## 学习目标

- 在命令行中浏览 Linux 文件系统并执行基本的文件操作
- 使用 `chmod` 和 `chown` 管理文件权限，解决“权限被拒绝”错误
- 使用 `apt` 安装系统软件包，并为 AI 工作配置一台全新的 GPU 机器
- 识别 macOS 与 Linux 之间的差异，这些差异常常使在远程机器上工作的开发者陷入困境

## 问题

你在 macOS 或 Windows 上开发。但当你通过 SSH 连接到云 GPU 机器、租用 Lambda 实例或启动一台 EC2 机器时，你面对的是 Ubuntu。终端是你唯一的界面。没有 Finder，没有资源管理器，没有图形界面。如果你不能从命令行浏览文件系统、安装软件包和管理进程，你就会在空闲的 GPU 小时上白白花钱，同时还在谷歌搜索“如何在 Linux 中解压文件”。

这是一份生存指南。它涵盖了你在一台远程 Linux 机器上进行 AI 工作所需的全部内容，不多不少。

## 文件系统布局

Linux 将所有内容组织在单一的根目录 `/` 下。没有 `C:\` 或 `/Volumes`。你实际会接触到的目录：

```mermaid
graph TD
    root["/"] --> home["home/your-username/<br/>Your files — clone repos, run training"]
    root --> tmp["tmp/<br/>Temporary files, cleared on reboot"]
    root --> usr["usr/<br/>System programs and libraries"]
    root --> etc["etc/<br/>Config files"]
    root --> varlog["var/log/<br/>Logs — check when something breaks"]
    root --> mnt["mnt/ or /media/<br/>External drives and volumes"]
    root --> proc["proc/ and /sys/<br/>Virtual files — kernel and hardware info"]
```

你的主目录是 `~` 或 `/home/你的用户名`。你做的几乎所有工作都在这里。

## 基本命令

下面这 15 条命令覆盖了你在远程 GPU 机器上 95% 的操作。

### 移动与导航

```bash
pwd                         # Where am I?
ls                          # What's here?
ls -la                      # What's here, including hidden files with details?
cd /path/to/dir             # Go there
cd ~                        # Go home
cd ..                       # Go up one level
```

### 文件和目录

```bash
mkdir my-project            # Create a directory
mkdir -p a/b/c              # Create nested directories in one shot

cp file.txt backup.txt      # Copy a file
cp -r src/ src-backup/      # Copy a directory (recursive)

mv old.txt new.txt          # Rename a file
mv file.txt /tmp/           # Move a file

rm file.txt                 # Delete a file (no trash, it's gone)
rm -rf my-dir/              # Delete a directory and everything inside
```

`rm -rf` 是永久性的，没有撤销操作。在按回车之前请仔细检查路径。

### 读取文件

```bash
cat file.txt                # Print entire file
head -20 file.txt           # First 20 lines
tail -20 file.txt           # Last 20 lines
tail -f log.txt             # Follow a log file in real time (Ctrl+C to stop)
less file.txt               # Scroll through a file (q to quit)
```

### 搜索

```bash
grep "error" training.log           # Find lines containing "error"
grep -r "learning_rate" .           # Search all files in current directory
grep -i "cuda" config.yaml          # Case-insensitive search

find . -name "*.py"                 # Find all Python files under current dir
find . -name "*.ckpt" -size +1G     # Find checkpoint files larger than 1GB
```

## 权限

Linux 中的每个文件都有所有者及权限位。当脚本无法执行或你无法写入某个目录时，你就会遇到这个问题。

```bash
ls -l train.py
# -rwxr-xr-- 1 user group 2048 Mar 19 10:00 train.py
#  ^^^             owner permissions: read, write, execute
#     ^^^          group permissions: read, execute
#        ^^        everyone else: read only
```

常见的修正方法：

```bash
chmod +x train.sh           # Make a script executable
chmod 755 deploy.sh         # Owner: full, others: read+execute
chmod 644 config.yaml       # Owner: read+write, others: read only

chown user:group file.txt   # Change who owns a file (needs sudo)
```

当系统提示“权限被拒绝”时，几乎总是权限问题。`chmod +x` 或 `sudo` 能解决大部分情况。

## 软件包管理（apt）

Ubuntu 使用 `apt`。这是你安装系统级软件的方式。

```bash
sudo apt update             # Refresh the package list (always do this first)
sudo apt install -y htop    # Install a package (-y skips confirmation)
sudo apt install -y build-essential  # C compiler, make, etc. Needed by many Python packages
sudo apt install -y tmux    # Terminal multiplexer (keep sessions alive after disconnect)

apt list --installed        # What's installed?
sudo apt remove htop        # Uninstall
```

在一台全新的 GPU 机器上你会安装的常见软件包：

```bash
sudo apt update && sudo apt install -y \
    build-essential \
    git \
    curl \
    wget \
    tmux \
    htop \
    unzip \
    python3-venv
```

## 用户与 sudo

你通常以普通用户身份登录。某些操作需要 root（管理员）权限。

```bash
whoami                      # What user am I?
sudo command                # Run a single command as root
sudo su                     # Become root (exit to go back, use sparingly)
```

在云 GPU 实例上，你通常是唯一用户并且已经有 sudo 权限。不要什么都用 root 运行。只在需要时使用 sudo。

## 进程与 systemd

当你的训练挂起时，或者你需要检查正在运行什么时：

```bash
htop                        # Interactive process viewer (q to quit)
ps aux | grep python        # Find running Python processes
kill 12345                  # Gracefully stop process with PID 12345
kill -9 12345               # Force kill (use when graceful doesn't work)
nvidia-smi                  # GPU processes and memory usage
```

systemd 管理服务（后台守护进程）。如果你运行推理服务器时会用到它：

```bash
sudo systemctl start nginx          # Start a service
sudo systemctl stop nginx           # Stop it
sudo systemctl restart nginx        # Restart it
sudo systemctl status nginx         # Check if it's running
sudo systemctl enable nginx         # Start automatically on boot
```

## 磁盘空间

GPU 机器通常磁盘空间有限。模型和数据集会很快填满空间。

```bash
df -h                       # Disk usage for all mounted drives
df -h /home                 # Disk usage for /home specifically

du -sh *                    # Size of each item in current directory
du -sh ~/.cache             # Size of your cache (pip, huggingface models land here)
du -sh /data/checkpoints/   # Check how big your checkpoints are

# Find the biggest space hogs
du -h --max-depth=1 / 2>/dev/null | sort -hr | head -20
```

常见的省空间方法：

```bash
# Clear pip cache
pip cache purge

# Clear apt cache
sudo apt clean

# Remove old checkpoints you don't need
rm -rf checkpoints/epoch_01/ checkpoints/epoch_02/
```

## 网络

你需要从命令行下载模型、传输文件和访问 API。

```bash
# Download files
wget https://example.com/model.bin                   # Download a file
curl -O https://example.com/data.tar.gz              # Same thing with curl
curl -s https://api.example.com/health | python3 -m json.tool  # Hit an API, pretty-print JSON

# Transfer files between machines
scp model.bin user@remote:/data/                     # Copy file to remote machine
scp user@remote:/data/results.csv .                  # Copy file from remote to local
scp -r user@remote:/data/checkpoints/ ./local-dir/   # Copy directory

# Sync directories (faster than scp for large transfers, resumes on failure)
rsync -avz --progress ./data/ user@remote:/data/
rsync -avz --progress user@remote:/results/ ./results/
```

对于大文件，使用 `rsync` 而非 `scp`。它只传输发生变化的字节，并能处理连接中断的情况。

## tmux：保持会话活跃

当你通过 SSH 连接到远程机器时，合上笔记本电脑会终止你的训练运行。tmux 可以避免这种情况。

```bash
tmux new -s train           # Start a new session named "train"
# ... start your training, then:
# Ctrl+B, then D            # Detach (training keeps running)

tmux ls                     # List sessions
tmux attach -t train        # Reattach to session

# Inside tmux:
# Ctrl+B, then %            # Split pane vertically
# Ctrl+B, then "            # Split pane horizontally
# Ctrl+B, then arrow keys   # Switch between panes
```

始终在 tmux 中运行长时间的训练任务。始终如此。

## Windows 用户的 WSL2

如果你在使用 Windows，WSL2 可以让你在不双系统的情况下获得真正的 Linux 环境。

```bash
# In PowerShell (admin)
wsl --install -d Ubuntu-24.04

# After restart, open Ubuntu from Start menu
sudo apt update && sudo apt upgrade -y
```

WSL2 运行真正的 Linux 内核。本课中的所有内容在 WSL2 内部都可以使用。从 WSL 内部访问你的 Windows 文件，路径为 `/mnt/c/Users/你的用户名/`。

如果在 Windows 端安装了 NVIDIA 驱动，GPU 穿透功能即可工作。安装 Windows 版 NVIDIA 驱动（不是 Linux 版），然后在 WSL2 中就能使用 CUDA。

## 常见陷阱：从 macOS 到 Linux

如果你是从 macOS 过来的，这些地方会让你措手不及：

| macOS | Linux | 备注 |
|-------|-------|------|
| `brew install` | `sudo apt install` | 有时候软件包名称不同。`brew install htop` 与 `sudo apt install htop` 效果相同，但 `brew install readline` 与 `sudo apt install libreadline-dev` 则不同。 |
| `open file.txt` | `xdg-open file.txt` | 但在远程机器上不会有图形界面。请使用 `cat` 或 `less`。 |
| `pbcopy` / `pbpaste` | 不可用 | 通过 SSH 无法使用剪贴板管道。 |
| `~/.zshrc` | `~/.bashrc` | macOS 默认使用 zsh。大多数 Linux 服务器使用 bash。 |
| `/opt/homebrew/` | `/usr/bin/`、`/usr/local/bin/` | 二进制文件位于不同位置。 |
| `sed -i '' 's/a/b/' file` | `sed -i 's/a/b/' file` | macOS 的 sed 需要在 `-i` 后跟一个空字符串。Linux 则不需要。 |
| 不区分大小写的文件系统 | 区分大小写的文件系统 | `Model.py` 和 `model.py` 在 Linux 上是两个不同的文件。 |
| 行尾 `\n` | 行尾 `\n` | 相同。但 Windows 使用 `\r\n`，这会破坏 bash 脚本。运行 `dos2unix` 来修复。 |

## 快速参考卡

```
Navigation:     pwd, ls, cd, find
Files:          cp, mv, rm, mkdir, cat, head, tail, less
Search:         grep, find
Permissions:    chmod, chown, sudo
Packages:       apt update, apt install
Processes:      htop, ps, kill, nvidia-smi
Services:       systemctl start/stop/restart/status
Disk:           df -h, du -sh
Network:        curl, wget, scp, rsync
Sessions:       tmux new/attach/detach
```

## 练习

1. SSH 到任何 Linux 机器（或打开 WSL2），然后导航到你的主目录。创建一个项目文件夹，用 `touch` 在其中创建三个空文件，然后用 `ls -la` 列出它们。
2. 使用 apt 安装 `htop`，运行它，并找出哪个进程占用内存最多。
3. 启动一个 tmux 会话，在内部运行 `sleep 300`，分离会话，列出会话，然后重新连接。
4. 使用 `df -h` 检查可用磁盘空间，然后使用 `du -sh ~/.cache/*` 找出缓存中占用空间的内容。
5. 使用 `scp` 将一个文件从本机传输到远程机器，然后使用 `rsync` 进行同样的传输，比较两者的体验。
