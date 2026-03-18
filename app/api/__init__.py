from app.api.admin import metrics
from app.api.alerts import process_alerts
from app.api.expiry import list_expiry
from app.api.health import health
from app.api.scans import create_scan, get_scan, patch_scan

ROUTES = [create_scan, get_scan, patch_scan, list_expiry, process_alerts, health, metrics]
