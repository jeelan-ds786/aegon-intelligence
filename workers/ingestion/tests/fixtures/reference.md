# API reference

| Method | Path |
| --- | --- |
| GET | /health |

```python
response = client.get("/health")
assert response.status_code == 200
```