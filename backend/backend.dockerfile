FROM python:3.12

WORKDIR /app

# Poetry
## Install poetry (turn off virtual env so it can build lightly & use cache.)
RUN pip install poetry==1.4.2 && poetry config virtualenvs.create false
## Set Poetry to non-interactive mode to avoid user interaction during builds
ENV POETRY_NO_INTERACTION=1
## Specify a cache directory for Poetry to improve build speeds and manage cache effectively
ENV POETRY_CACHE_DIR=/tmp/poetry_cache
## Disable Python's buffering to ensure that logs are streamed to the console immediately
ENV PYTHONUNBUFFERED 1
## Prevent Python from writing bytecode files (.pyc) to reduce image size
ENV PYTHONDONTWRITEBYTECODE 1
