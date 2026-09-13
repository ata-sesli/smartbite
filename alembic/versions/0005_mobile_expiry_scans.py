"""add mobile expiry scans

Revision ID: 0005_mobile_expiry_scans
Revises: 0004_test64_expected_dates
Create Date: 2026-05-11 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0005_mobile_expiry_scans"
down_revision = "0004_test64_expected_dates"
branch_labels = None
depends_on = None


uuid_type = sa.Uuid(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "mobile_expiry_scans",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("image_blob", sa.LargeBinary(), nullable=False),
        sa.Column("image_filename", sa.String(length=255), nullable=False),
        sa.Column("image_content_type", sa.String(length=128), nullable=False),
        sa.Column("detected_expiry_date", sa.Date(), nullable=True),
        sa.Column("corrected_expiry_date", sa.Date(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("normalized_text", sa.Text(), nullable=True),
        sa.Column("recognition_confidence", sa.Float(), nullable=True),
        sa.Column("detector_confidence", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=64), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("detection_polygon_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("corrected_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_mobile_expiry_scans_status", "mobile_expiry_scans", ["status"])


def downgrade() -> None:
    op.drop_index("ix_mobile_expiry_scans_status", table_name="mobile_expiry_scans")
    op.drop_table("mobile_expiry_scans")
