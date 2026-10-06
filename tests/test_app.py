from fastapi.testclient import TestClient
from backend.main import app

client=TestClient(app)

def test_health():
    r=client.get('/api/health'); assert r.status_code==200; assert r.json()['status']=='ok'

def test_demo():
    r=client.get('/api/demo'); assert r.status_code==200; assert r.json()['profile']['rows']>0

def test_analytics_endpoints():
    assert client.get('/api/segmentation').status_code==200
    assert client.get('/api/forecast').status_code==200
    assert client.get('/api/insights').status_code==200
