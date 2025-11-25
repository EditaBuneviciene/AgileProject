# LLM Data Analysis App
## Quick Start
```bash
(Win)Download Docker Desktop from https://www.docker.com/products/docker-desktop/
Start Docker from desktop icon, if "Virtualization supported not detected" then virtualisation (SVM in AMD) in BIOS has to be enabled.
To check if docker running in cmd: 
docker --version
docker ps
Then in cmd/terminal go to project directory and build docker image
cd AgileProject/sprint3
docker build -t data-analysis-app .
Then run build image:
win command:
docker run -it data-analysis-app
docker run data-analysis-app python -m pytest test_main.py -v