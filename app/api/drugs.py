from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Drug
from app.schemas import DrugCreate, DrugResponse
from sqlalchemy.exc import IntegrityError
from app.models import Drug, DemandHistory

from datetime import timedelta

import pandas as pd

from app.services.feature_service import build_features
from app.model.model_loader import model, features

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

@router.post("/{drug_id}/forecast")
def forecast_drug(
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

    history_records = (
        db.query(DemandHistory)
        .filter(DemandHistory.drug_id == drug_id)
        .order_by(DemandHistory.date)
        .all()
    )

    if len(history_records) < 28:
        raise HTTPException(
            status_code=400,
            detail="At least 28 days of demand history are required"
        )

    history = pd.DataFrame([
        {
            "date": record.date,
            "demand": record.demand
        }
        for record in history_records
    ])

    last_date = history["date"].max()

    target_date = (
        pd.Timestamp(last_date)
        + timedelta(days=1)
    )

    X = build_features(
        history,
        target_date
    )

    X = X[features]

    prediction = model.predict(X)[0]

    return {
        "drug_id": drug.id,
        "drug_code": drug.code,
        "forecast_date": target_date.date(),
        "predicted_demand": float(prediction)
    }