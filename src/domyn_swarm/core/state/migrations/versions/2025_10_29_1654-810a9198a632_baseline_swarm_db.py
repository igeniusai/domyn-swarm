# Copyright 2025 iGenius S.p.A
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""baseline swarm db

Revision ID: 810a9198a632
Revises:
Create Date: 2025-10-29 16:54:25.894931

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = "810a9198a632"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # If the table does not exist, create it from scratch
    bind = op.get_bind()
    inspector = inspect(bind)

    if not inspector.has_table("swarm"):
        op.create_table(
            "swarm",
            sa.Column("deployment_name", sa.String(), primary_key=True),
            sa.Column("swarm", sa.JSON(), nullable=False),
            sa.Column("cfg", sa.JSON(), nullable=False),
            sa.Column("serving_handle", sa.JSON(), nullable=False),
            sa.Column(
                "creation_dt",
                sa.DateTime(timezone=False),
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                nullable=False,
            ),
        )
        return
    else:
        # If the table exists, we assume it was created with older types (TEXT
        # instead of JSON, nullable columns, etc.); we alter the columns to
        # match the current model.
        with op.batch_alter_table("swarm") as batch_op:
            batch_op.alter_column(
                "deployment_name",
                existing_type=sa.TEXT(),
                type_=sa.String(),
                nullable=False,
            )
            batch_op.alter_column(
                "swarm", existing_type=sa.TEXT(), type_=sa.JSON(), nullable=False
            )
            batch_op.alter_column(
                "cfg", existing_type=sa.TEXT(), type_=sa.JSON(), nullable=False
            )
            batch_op.alter_column(
                "serving_handle",
                existing_type=sa.TEXT(),
                type_=sa.JSON(),
                nullable=False,
            )
            batch_op.alter_column(
                "creation_dt",
                existing_type=sa.DATETIME(),
                nullable=False,
                server_default=sa.text("(CURRENT_TIMESTAMP)"),
                existing_server_default=sa.text("(CURRENT_TIMESTAMP)"),
            )
    # ### end Alembic commands ###


def downgrade() -> None:
    """Drop the local swarm state table on downgrade.

    This is safe because swarm.db is local, ephemeral state.
    On next CLI invocation, the DB will be recreated and
    migrated to whatever the current head is.
    """
    bind = op.get_bind()
    inspector = inspect(bind)

    if inspector.has_table("swarm"):
        op.drop_table("swarm")
