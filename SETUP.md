# 🛠️ Team Setup Guide — AI Sprint Manager OpenEnv

Complete setup instructions for Windows and Mac teammates.

---

## 📋 Prerequisites

### Windows

1. **Install Python 3.11**
   - Download from https://www.python.org/downloads/
   - ✅ Check **"Add Python to PATH"** during install
   - Verify: open PowerShell and run `python --version`

2. **Install Git**
   - Download from https://git-scm.com/download/win
   - Use default options during install
   - Verify: `git --version`

3. **Install Docker Desktop**
   - Download from https://www.docker.com/products/docker-desktop/
   - Requires Windows 10/11 with WSL2 enabled
   - After install, open Docker Desktop and wait for it to start
   - Verify: `docker --version`

4. **Install VS Code** (recommended)
   - Download from https://code.visualstudio.com/

### Mac

1. **Install Homebrew** (package manager)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python 3.11**
   ```bash
   brew install python@3.11
   ```
   Verify: `python3.11 --version`

3. **Install Git**
   ```bash
   brew install git
   ```
   Verify: `git --version`

4. **Install Docker Desktop**
   - Download from https://www.docker.com/products/docker-desktop/
   - Choose Apple Silicon or Intel depending on your Mac
   - Open Docker Desktop and wait for it to start
   - Verify: `docker --version`

---

## 📥 Clone the GitHub Repo

```bash
# Both Windows (PowerShell) and Mac (Terminal)
git clone https://github.com/YOUR_GITHUB_USERNAME/ai-sprint-manager-openenv.git
cd ai-sprint-manager-openenv
```

---

## 🌿 Create Your Own Branch

```bash
# Replace "yourname" with your actual name
git checkout -b feature/yourname-improvements
```

Verify you're on your branch:
```bash
git branch
# Should show * feature/yourname-improvements
```

---

## 🐍 Set Up Virtual Environment

### Windows
```powershell
python -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
pip install python-dotenv
```

### Mac
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install python-dotenv
```

---

## ⚙️ Create Your `.env` File

Create a file called `.env` in the project root (never commit this!):

```
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
ENV_BASE_URL=http://localhost:7860
```

We'll get the HF_TOKEN in the next section.

---

## 🤗 Create Hugging Face Account & Token

1. Go to https://huggingface.co and sign up
2. Go to https://huggingface.co/settings/tokens
3. Click **New token**
4. Name: `sprint-manager-token`
5. Type: **Read**
6. Click **Create token** and copy it
7. Paste it into your `.env` file as `HF_TOKEN=hf_...`

---

## 🚀 Run Locally & Test

### Terminal 1 — Start the API + UI server
#### Windows
```powershell
python ui.py
```
#### Mac
```bash
python ui.py
```

### Terminal 2 — Test the API
#### Windows
```powershell
Invoke-WebRequest -Uri http://localhost:7860/health -Method GET
```
#### Mac
```bash
curl http://localhost:7860/health
# Expected: {"status":"ok","env":"ai-sprint-manager"}
```

### Test via Browser
Open http://localhost:7860 — you should see the Gradio sprint board UI.

Try:
1. Select `easy_sprint`, click **🔄 Reset Sprint**
2. Set Action=`assign`, Task ID=`T1`, Dev ID=`dev1`
3. Click **▶️ Take Action**
4. You should see a positive reward and T1 assigned to Alice

### Run Inference Locally
```bash
# Make sure .env file has your HF_TOKEN
python inference.py
```

---

## 🐳 Test with Docker

```bash
docker build -t ai-sprint-manager .
```

### Windows
```powershell
docker run -p 7860:7860 ai-sprint-manager
```
### Mac
```bash
docker run -p 7860:7860 ai-sprint-manager
```

Open http://localhost:7860 to verify.

---

## ☁️ Create Your Own HF Space & Deploy

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name:** `ai-sprint-manager-yourname`
   - **SDK:** Docker
   - **Visibility:** Public
3. Click **Create Space**

4. Add your HF token as a Secret:
   - Go to Space → **Settings** → **Variables and secrets**
   - Add secret: Name=`HF_TOKEN`, Value=your token

5. Add HF as a git remote and push your branch:
```bash
git remote add myspace https://huggingface.co/spaces/YOUR_HF_USERNAME/ai-sprint-manager-yourname
git push myspace feature/yourname-improvements:main
```

6. Wait 2-3 minutes for build. Test your live Space:

### Windows
```powershell
Invoke-WebRequest -Uri "https://YOUR_HF_USERNAME-ai-sprint-manager-yourname.hf.space/health" -Method GET
```
### Mac
```bash
curl https://YOUR_HF_USERNAME-ai-sprint-manager-yourname.hf.space/health
```

---

## 🔧 Run OpenEnv Validation

```bash
pip install openenv-core uv
uv lock
openenv validate
# Expected: [OK] ai-sprint-manager: Ready for multi-mode deployment
```

---

## 💡 Suggested Improvements for Teammates

### 🟢 Easy (Good for getting started)

1. **Better skill matching UI** — Show a skill-to-dev mapping table in the Gradio UI so users know which dev to pick
2. **Sprint history chart** — Add a reward-over-time line chart using `gr.Plot`
3. **Add more tasks** — Expand `tasks.py` with more realistic task names and types
4. **Auto-assign button** — Add a Gradio button that automatically assigns all backlog tasks using a simple rule (best skill match)

### 🟡 Medium

5. **Session isolation** — Currently all users share one env instance. Add session IDs so multiple users can use the UI simultaneously
6. **Sprint replay** — Save the full episode history and add a "replay" feature to the UI
7. **Better reward visualization** — Add a bar chart showing per-task completion status
8. **Configurable sprint length** — Let users set sprint length (5/10/15 days) in the UI

### 🔴 Hard (Advanced)

9. **Real RL training loop** — Add a `train.py` script using Stable-Baselines3 or TRL+GRPO to actually train a policy
10. **Multi-agent support** — Let multiple AI agents collaborate on sprint management
11. **WebSocket support** — Upgrade from HTTP to WebSocket for real-time updates per OpenEnv spec
12. **Custom sprint builder** — Let users define their own tasks and team in the UI

---

## 📤 Submit Your Changes

```bash
git add .
git commit -m "feat: your improvement description"
git push origin feature/yourname-improvements
```

Then open a Pull Request on GitHub to merge into `main`.