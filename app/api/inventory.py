from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from app.database import get_db
from app.models import Drug, Inventory
from app.schemas import InventoryCreate, InventoryResponse

router = APIRouter(
    prefix="/inventory",
    tags=["inventory"],
)

@router.post(
     "",
    response_model=InventoryResponse,
    status_code=201
)
def create_inventory(
        data: InventoryCreate,
        db: Session = Depends(get_db),
):
    drug = (
        db.query(Drug)
        .filter(Drug.id == data.drug_id)
        .first()
    )

    if drug is None:
        raise HTTPException(
            status_code=404,
            detail="Drug not found"
        )

    inventory = Inventory(
        drug_id=data.drug_id,
        current_stock=data.current_stock,
        safety_stock=data.safety_stock,
        lead_time_days=data.lead_time_days
    )

    try:
        db.add(inventory)
        db.commit()
        db.refresh(inventory)

    except IntegrityError:
        db.rollback()

        raise HTTPException(
            status_code=409,
            detail="Inventory already exists for this drug"
        )

    return inventory


@router.get(
    "",
    response_model=list[InventoryResponse]
)
def get_inventory(
    db: Session = Depends(get_db)
):
    return db.query(Inventory).all()