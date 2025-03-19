from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

# User Table
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)

    # Relationships
    subscriptions = relationship("UserSubscription", back_populates="user")

# Subscription Plan Table
class SubscriptionPlan(Base):
    __tablename__ = "subscription_plans"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    price = Column(Integer, nullable=False)
    description = Column(String)
    duration_in_days = Column(Integer, nullable=False)

    # Relationships
    user_subscriptions = relationship("UserSubscription", back_populates="subscription_plan")

# User-Subscription Relationship Table
class UserSubscription(Base):
    __tablename__ = "user_subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    subscription_id = Column(Integer, ForeignKey("subscription_plans.id"), nullable=False)
    start_date = Column(DateTime(timezone=True), server_default=func.now())
    end_date = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="subscriptions")
    subscription_plan = relationship("SubscriptionPlan", back_populates="user_subscriptions")
