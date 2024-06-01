# RAG_from_scratch
The following are required:
- docker
- Makefile

<br>

# How to develop backend in vscode.
## 1. Find poetry virtual env path
```shell
poetry env info --path
```
## 2. set virtualenv path to vscode global settings.
- add poetry env path to vscode `python.pythonPath`.
  - `python.pythonPath` is at setting - search `venv`
reference: https://stackoverflow.com/questions/59882884/vscode-doesnt-show-poetry-virtualenvs-in-select-interpreter-option

<br>

# How to run
## 1. Create `.env.dev`
- 1. Refer to .env.template to create `.env.dev` in the root directory.
## 2. Build the images.
There are two methods for building images: the basic way and the Makefile way.
### A. Basic
- In ./backend,
```shell
docker build -t backend-dev:python -f backend.dockerfile .
```
### B. Makefile
- In root dir
```shell
make build_backend
```
## 3. Create docker network
- default network name is `main-bridge`
``` shell
docker network create main-bridge
```
## 4. compose up
There are two methods to launch the compose: the basic way and the Makefile way.
### A. Basic
- In root dir
```shell
docker compose --env-file .env.dev -f docker-compose.backend.dev.yml up -d
```
### B. Makefile
- In root dir
```shell
make run_backend
make down_backend
```

<br>

# Default port
## backend
- 8000:8000

<br>
