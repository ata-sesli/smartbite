"""add product text results

Revision ID: 0003_product_text_results
Revises: 0002_scan_reviews
Create Date: 2026-05-01 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0003_product_text_results"
down_revision = "0002_scan_reviews"
branch_labels = None
depends_on = None


uuid_type = sa.Uuid(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "product_text_results",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=True),
        sa.Column("product_name", sa.String(length=255), nullable=True),
        sa.Column("brand", sa.String(length=255), nullable=True),
        sa.Column("full_text", sa.Text(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("engine_name", sa.String(length=64), nullable=False),
        sa.Column("runtime_device", sa.String(length=16), nullable=True),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("source_endpoint", sa.String(length=128), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_product_text_results_scan_id", "product_text_results", ["scan_id"])


def downgrade() -> None:
    op.drop_index("ix_product_text_results_scan_id", table_name="product_text_results")
    op.drop_table("product_text_results")
