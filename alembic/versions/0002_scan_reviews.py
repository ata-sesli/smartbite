"""add scan reviews

Revision ID: 0002_scan_reviews
Revises: 0001_initial
Create Date: 2026-04-18 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0002_scan_reviews"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


uuid_type = sa.Uuid(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "scan_reviews",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("scan_id", uuid_type, sa.ForeignKey("scans.id"), nullable=False),
        sa.Column("verdict", sa.String(length=64), nullable=False),
        sa.Column("accepted_for_training", sa.Boolean(), nullable=False),
        sa.Column("final_text", sa.Text(), nullable=True),
        sa.Column("final_parsed_date", sa.Date(), nullable=True),
        sa.Column("bbox_xyxy", sa.JSON(), nullable=True),
        sa.Column("bbox_source", sa.String(length=64), nullable=False),
        sa.Column("reviewer_id", sa.String(length=128), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("scan_id", name="uq_scan_reviews_scan_id"),
    )
    op.create_index("ix_scan_reviews_scan_id", "scan_reviews", ["scan_id"])
    op.create_index("ix_scan_reviews_verdict", "scan_reviews", ["verdict"])
    op.create_index("ix_scan_reviews_accepted_for_training", "scan_reviews", ["accepted_for_training"])
    op.create_index("ix_scan_reviews_reviewer_id", "scan_reviews", ["reviewer_id"])


def downgrade() -> None:
    op.drop_index("ix_scan_reviews_reviewer_id", table_name="scan_reviews")
    op.drop_index("ix_scan_reviews_accepted_for_training", table_name="scan_reviews")
    op.drop_index("ix_scan_reviews_verdict", table_name="scan_reviews")
    op.drop_index("ix_scan_reviews_scan_id", table_name="scan_reviews")
    op.drop_table("scan_reviews")
