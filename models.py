from sqlalchemy import Column, Integer, String, Float, Date
from database import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    type = Column(String(20), nullable=False)  # income / expense
    category = Column(String(50), nullable=True)
    description = Column(String(255), nullable=True)
    amount = Column(Float, nullable=False)
    customer = Column(String(100), nullable=True)
    product = Column(String(100), nullable=True)
