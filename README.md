# Stock Prediction API Container

## Build

From the **project root** (where `app/`, `container/` are located):

```bash
docker build -f container/Dockerfile -t stock-api .
```


## Run

### With artifacts mounted at runtime (recommended)

```bash
docker run -p 8000:8000 -v $(pwd)/model_artifacts:/app/model_artifacts stock-api
```

### With artifacts already in the image

```bash
docker run -p 8000:8000 stock-api
```

API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.
