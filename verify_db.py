"""
Verify database schema and connectivity.
Updated for the new 3-model schema (Prediction, AuditLog, RequestMetric).
"""
from app import app, db, Prediction, AuditLog, RequestMetric


def verify():
    with app.app_context():
        db.create_all()
        print("✅ Database tables created (or already exist).")

        pred_count = Prediction.query.count()
        print(f"   Predictions: {pred_count}")

        audit_count = AuditLog.query.count()
        print(f"   Audit logs:  {audit_count}")

        metric_count = RequestMetric.query.count()
        print(f"   Metrics:     {metric_count}")

        print("✅ Verification successful!")


if __name__ == "__main__":
    verify()
