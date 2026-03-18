"""initial schema

Revision ID: 0001_initial
Revises: 
Create Date: 2026-03-18 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


uuid_type = sa.Uuid(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("qr_code", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.Column("brand", sa.String(length=255), nullable=True),
        sa.Column("category", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("qr_code", name="uq_products_qr_code"),
    )
    op.create_index("ix_products_qr_code", "products", ["qr_code"])

    op.create_table(
        "scans",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("product_id", uuid_type, sa.ForeignKey("products.id"), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("image_path", sa.String(length=500), nullable=False),
        sa.Column("roi_path", sa.String(length=500), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("final_result_status", sa.String(length=64), nullable=True),
        sa.Column("detector_confidence", sa.Float(), nullable=True),
        sa.Column("detected", sa.Boolean(), nullable=True),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_scans_product_id", "scans", ["product_id"])
    op.create_index("ix_scans_user_id", "scans", ["user_id"])
    op.create_index("ix_scans_status", "scans", ["status"])
    op.create_index("ix_scans_final_result_status", "scans", ["final_result_status"])

    op.create_table(
        "ocr_results",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("raw_text", sa.Text(), nullable=False),
        sa.Column("normalized_text", sa.Text(), nullable=False),
        sa.Column("ocr_confidence", sa.Float(), nullable=True),
        sa.Column("engine_name", sa.String(length=64), nullable=False),
        sa.Column("runtime_device", sa.String(length=16), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("scan_id", name="uq_ocr_results_scan_id"),
    )
    op.create_index("ix_ocr_results_scan_id", "ocr_results", ["scan_id"])

    op.create_table(
        "parsed_date_results",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("parsed_date", sa.Date(), nullable=True),
        sa.Column("date_format_detected", sa.String(length=64), nullable=True),
        sa.Column("parse_confidence", sa.Float(), nullable=True),
        sa.Column("parser_reason", sa.Text(), nullable=False),
        sa.Column("candidate_dates_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("scan_id", name="uq_parsed_date_results_scan_id"),
    )
    op.create_index("ix_parsed_date_results_scan_id", "parsed_date_results", ["scan_id"])

    op.create_table(
        "expiry_statuses",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("days_remaining", sa.Integer(), nullable=True),
        sa.Column("alert_required", sa.Boolean(), nullable=False),
        sa.Column("needs_review", sa.Boolean(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("scan_id", name="uq_expiry_statuses_scan_id"),
    )
    op.create_index("ix_expiry_statuses_scan_id", "expiry_statuses", ["scan_id"])
    op.create_index("ix_expiry_statuses_status", "expiry_statuses", ["status"])

    op.create_table(
        "alert_events",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("user_id", sa.String(length=128), nullable=False),
        sa.Column("alert_type", sa.String(length=64), nullable=False),
        sa.Column("delivery_status", sa.String(length=32), nullable=False),
        sa.Column("payload_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("scan_id", "alert_type", name="uq_alert_events_scan_type"),
    )
    op.create_index("ix_alert_events_scan_id", "alert_events", ["scan_id"])
    op.create_index("ix_alert_events_user_id", "alert_events", ["user_id"])
    op.create_index("ix_alert_events_alert_type", "alert_events", ["alert_type"])
    op.create_index("ix_alert_events_delivery_status", "alert_events", ["delivery_status"])

    op.create_table(
        "processing_logs",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("level", sa.String(length=16), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("elapsed_ms", sa.Integer(), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_processing_logs_scan_id", "processing_logs", ["scan_id"])
    op.create_index("ix_processing_logs_stage", "processing_logs", ["stage"])


def downgrade() -> None:
    op.drop_index("ix_processing_logs_stage", table_name="processing_logs")
    op.drop_index("ix_processing_logs_scan_id", table_name="processing_logs")
    op.drop_table("processing_logs")

    op.drop_index("ix_alert_events_delivery_status", table_name="alert_events")
    op.drop_index("ix_alert_events_alert_type", table_name="alert_events")
    op.drop_index("ix_alert_events_user_id", table_name="alert_events")
    op.drop_index("ix_alert_events_scan_id", table_name="alert_events")
    op.drop_table("alert_events")

    op.drop_index("ix_expiry_statuses_status", table_name="expiry_statuses")
    op.drop_index("ix_expiry_statuses_scan_id", table_name="expiry_statuses")
    op.drop_table("expiry_statuses")

    op.drop_index("ix_parsed_date_results_scan_id", table_name="parsed_date_results")
    op.drop_table("parsed_date_results")

    op.drop_index("ix_ocr_results_scan_id", table_name="ocr_results")
    op.drop_table("ocr_results")

    op.drop_index("ix_scans_final_result_status", table_name="scans")
    op.drop_index("ix_scans_status", table_name="scans")
    op.drop_index("ix_scans_user_id", table_name="scans")
    op.drop_index("ix_scans_product_id", table_name="scans")
    op.drop_table("scans")

    op.drop_index("ix_products_qr_code", table_name="products")
    op.drop_table("products")
