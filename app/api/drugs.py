from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Drug
from app.schemas import DrugCreate, DrugResponse
from sqlalchemy.exc import IntegrityError
from app.models import Drug, DemandHistory

from app.schemas import (
    DrugCreate,
    DrugResponse,
    DemandHistoryResponse,
)


router = APIRouter(
    prefix="/drugs",
    tags=["drugs"]
)

@router.post("", response_model=DrugResponse)
def create_drug(
    drug: DrugCreate,
    db: Session = Depends(get_db)
):
    db_drug = Drug(
        code=drug.code,
        name=drug.name
    )

    try:
        db.add(db_drug)
        db.commit()
        db.refresh(db_drug)
    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=409,
            detail=f"Drug with code {drug.code} already exists"
        )

    return db_drug

@router.get("", response_model=list[DrugResponse])
def get_drugs(
    db: Session = Depends(get_db)
):
    drugs = db.query(Drug).all()

    return drugs

@router.get(
    "/{drug_id}/demand-history",
    response_model=list[DemandHistoryResponse]
)
def get_demand_history(
    drug_id: int,
    db: Session = Depends(get_db)
):
    drug = (
        db.query(Drug)
        .filter(Drug.id == drug_id)
        .first()
    )

    if drug is None:
        raise HTTPException(
            status_code=404,
            detail="Drug not found"
        )

    history = (
        db.query(DemandHistory)
        .filter(DemandHistory.drug_id == drug_id)
        .order_by(DemandHistory.date)
        .all()
    )

    return history
