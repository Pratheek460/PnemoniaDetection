from sqlalchemy import *

from sqlalchemy.orm import *

Base = declarative_base()

engine = create_engine(
    "sqlite:///database/pneumonia.db"
)

SessionLocal = sessionmaker(
    bind=engine
)

class Scan(Base):

    __tablename__ = "scans"

    id = Column(
        Integer,
        primary_key=True
    )

    patient_name = Column(
        String
    )

    prediction = Column(
        String
    )

    confidence = Column(
        Float
    )

    image_path = Column(
        String
    )
    
    heatmap_path=Column(
        String
    )

Base.metadata.create_all(engine)