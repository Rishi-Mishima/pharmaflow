import pandas as pd

from app.database import SessionLocal
from app.models import Drug, DemandHistory


def seed_database():
    db = SessionLocal()

    try:
        # 1. Find N02BE drug
        drug = (
            db.query(Drug)
            .filter(Drug.code == "N02BE")
            .first()
        )

        # 2. Create it if it doesn't exist
        if drug is None:
            drug = Drug(
                code="N02BE",
                name="Paracetamol"
            )

            db.add(drug)
            db.commit()
            db.refresh(drug)

        # 3. Read historical demand data
        df = pd.read_csv(
            "data/processed/n02be_daily.csv"
        )

        df["datum"] = pd.to_datetime(df["datum"])

        # 4. Convert each row into a DemandHistory object
        records = []

        for _, row in df.iterrows():
            record = DemandHistory(
                drug_id=drug.id,
                date=row["datum"].date(),
                demand=float(row["demand"])
            )

            records.append(record)

        # 5. Insert into PostgreSQL
        db.add_all(records)
        db.commit()

        print(
            f"Inserted {len(records)} demand records "
            f"for {drug.code}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    seed_database()