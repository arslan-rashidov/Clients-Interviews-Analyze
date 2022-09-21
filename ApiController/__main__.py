import uvicorn

uvicorn.run(
    'ApiController.app:app',
    host='127.0.0.1',
    reload=True,
    port=8000
)
