from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base

from datetime import date

from sqlalchemy import Date, Float, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column


class Drug(Base):
    __tablename__ = "drugs"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(100))


class DemandHistory(Base):
    __tablename__ = "demand_history"

    id: Mapped[int] = mapped_column(primary_key=True)

    drug_id: Mapped[int] = mapped_column(
        ForeignKey("drugs.id"),
        index=True
    )

    date: Mapped[date] = mapped_column(
        Date,
        index=True
    )

    demand: Mapped[float] = mapped_column(Float)