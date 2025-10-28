# Старт
```commandline
docker build -t app-image .
docker run -d -p 8501:8501 --name app-container app-image

```
# Завершение
```commandline
docker stop app-container
docker rm app-container
docker rmi app-image
```