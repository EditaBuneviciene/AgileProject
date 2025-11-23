# LLM Data Analysis App

## Quick Start
```bash
(Win)Download Docker Desktop from https://www.docker.com/products/docker-desktop/
cd AgileProject/sprint3
docker build -t data-analysis-app .
docker run -it -v $(pwd)/data:/app/data data-analysis-app