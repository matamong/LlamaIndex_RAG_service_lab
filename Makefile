ifeq ($(OS),Windows_NT)     # is Windows_NT on XP, 2000, 7, Vista, 10...
    detected_OS := Windows
	DOCKER_PREFIX = docker
else
    detected_OS := $(shell uname)  # same as "uname -s"
	DOCKER_PREFIX = sudo docker
endif

build_backend:
	@cd backend && $(DOCKER_PREFIX) build -t backend-dev -f backend.dockerfile .
	@echo "Backend image built"

run_backend:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.backend.dev.yml up

down_backend:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.backend.dev.yml down

run_others:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.dev.yml up -d

down_others:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.dev.yml down

run_all:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.dev.yml up -d
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.backend.dev.yml up

down_all:
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.dev.yml down
	$(DOCKER_PREFIX) compose --env-file .env.dev -f docker-compose.backend.dev.yml down
