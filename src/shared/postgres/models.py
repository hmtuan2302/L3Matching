from __future__ import annotations

from datetime import datetime
from typing import Dict
from typing import Optional

from shared.pg.database import Role
from sqlalchemy import Enum
from sqlalchemy import ForeignKey
from sqlalchemy import func
from sqlalchemy import Index
from sqlalchemy import String
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship


class Base(DeclarativeBase):
    pass


class Dated(Base):
    __abstract__ = True

    createdAt: Mapped[datetime] = mapped_column(insert_default=func.now())
    updatedAt: Mapped[datetime] = mapped_column(onupdate=func.now(), nullable=True)
    deletedAt: Mapped[datetime] = mapped_column(nullable=True)


class Identified(Base):
    __abstract__ = True

    id: Mapped[str] = mapped_column(primary_key=True, index=True)


class Channel(Identified, Dated):
    __tablename__ = 'channel'

    channel_name: Mapped[str]
    is_archived: Mapped[bool]
    is_general: Mapped[bool]
    is_shared: Mapped[bool]
    is_org_shared: Mapped[bool]
    is_member: Mapped[bool]
    is_private: Mapped[bool]
    is_mpim: Mapped[bool]
    latest: Mapped[str]
    topic: Mapped[str]
    purpose: Mapped[str]