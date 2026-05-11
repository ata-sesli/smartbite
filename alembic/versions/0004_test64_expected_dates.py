"""add test64 expected dates

Revision ID: 0004_test64_expected_dates
Revises: 0003_product_text_results
Create Date: 2026-05-02 00:00:00.000000
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "0004_test64_expected_dates"
down_revision = "0003_product_text_results"
branch_labels = None
depends_on = None


uuid_type = sa.Uuid(as_uuid=True)


def upgrade() -> None:
    op.create_table(
        "test64_expected_dates",
        sa.Column("id", uuid_type, primary_key=True, nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("expected_day", sa.Integer(), nullable=True),
        sa.Column("expected_month", sa.Integer(), nullable=False),
        sa.Column("expected_year", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("filename", name="uq_test64_expected_dates_filename"),
    )
    op.create_index("ix_test64_expected_dates_filename", "test64_expected_dates", ["filename"])


def downgrade() -> None:
    op.drop_index("ix_test64_expected_dates_filename", table_name="test64_expected_dates")
    op.drop_table("test64_expected_dates")
