"""SQLite database for historical data and trade records."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class Trade(Base):
    """Trade record model."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    pair_asset1 = Column(String(10), nullable=False)
    pair_asset2 = Column(String(10), nullable=False)
    signal = Column(String(20), nullable=False)
    zscore = Column(Float)
    hedge_ratio = Column(Float)
    shares_asset1 = Column(Integer)
    shares_asset2 = Column(Integer)
    price_asset1 = Column(Float)
    price_asset2 = Column(Float)
    spread_value = Column(Float)
    account_value = Column(Float)
    notes = Column(Text)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "pair": (self.pair_asset1, self.pair_asset2),
            "signal": self.signal,
            "zscore": self.zscore,
            "hedge_ratio": self.hedge_ratio,
            "shares": {"asset1": self.shares_asset1, "asset2": self.shares_asset2},
            "prices": {"asset1": self.price_asset1, "asset2": self.price_asset2},
            "spread_value": self.spread_value,
            "account_value": self.account_value,
            "notes": self.notes,
        }


class CointegrationResult(Base):
    """Cointegration test result model."""

    __tablename__ = "cointegration_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset1 = Column(String(10), nullable=False)
    asset2 = Column(String(10), nullable=False)
    p_value = Column(Float)
    adf_statistic = Column(Float)
    half_life = Column(Float)
    beta = Column(Float)
    is_cointegrated = Column(Integer)  # SQLite doesn't have boolean
    lookback_days = Column(Integer)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "pair": (self.asset1, self.asset2),
            "p_value": self.p_value,
            "adf_statistic": self.adf_statistic,
            "half_life": self.half_life,
            "beta": self.beta,
            "is_cointegrated": bool(self.is_cointegrated),
            "lookback_days": self.lookback_days,
        }


class DailyPnL(Base):
    """Daily P&L record model."""

    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True)
    starting_equity = Column(Float)
    ending_equity = Column(Float)
    realized_pnl = Column(Float)
    unrealized_pnl = Column(Float)
    total_pnl = Column(Float)
    num_trades = Column(Integer)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "date": self.date,
            "starting_equity": self.starting_equity,
            "ending_equity": self.ending_equity,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "num_trades": self.num_trades,
        }


class Database:
    """Database manager for the trading bot."""

    def __init__(self, db_path: str = "data/trading.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{db_path}")
        Base.metadata.create_all(self.engine)

        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Initialized database at {db_path}")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    def record_trade(
        self,
        pair: tuple[str, str],
        signal: str,
        zscore: float,
        hedge_ratio: float,
        shares: tuple[int, int],
        prices: tuple[float, float],
        spread_value: float,
        account_value: float,
        notes: Optional[str] = None,
    ) -> Trade:
        """
        Record a trade to the database.

        Args:
            pair: Asset pair tuple
            signal: Trade signal
            zscore: Z-score at trade time
            hedge_ratio: Hedge ratio used
            shares: Tuple of (shares_asset1, shares_asset2)
            prices: Tuple of (price_asset1, price_asset2)
            spread_value: Spread value at trade time
            account_value: Account value at trade time
            notes: Optional trade notes

        Returns:
            Created Trade object
        """
        with self.get_session() as session:
            trade = Trade(
                pair_asset1=pair[0],
                pair_asset2=pair[1],
                signal=signal,
                zscore=zscore,
                hedge_ratio=hedge_ratio,
                shares_asset1=shares[0],
                shares_asset2=shares[1],
                price_asset1=prices[0],
                price_asset2=prices[1],
                spread_value=spread_value,
                account_value=account_value,
                notes=notes,
            )
            session.add(trade)
            session.commit()
            session.refresh(trade)

            logger.info(f"Recorded trade: {signal} for {pair}")
            return trade

    def record_cointegration_result(
        self,
        pair: tuple[str, str],
        p_value: float,
        adf_statistic: float,
        half_life: float,
        beta: float,
        is_cointegrated: bool,
        lookback_days: int,
    ) -> CointegrationResult:
        """Record cointegration test result."""
        with self.get_session() as session:
            result = CointegrationResult(
                asset1=pair[0],
                asset2=pair[1],
                p_value=p_value,
                adf_statistic=adf_statistic,
                half_life=half_life,
                beta=beta,
                is_cointegrated=int(is_cointegrated),
                lookback_days=lookback_days,
            )
            session.add(result)
            session.commit()
            session.refresh(result)
            return result

    def record_daily_pnl(
        self,
        date: str,
        starting_equity: float,
        ending_equity: float,
        realized_pnl: float,
        unrealized_pnl: float,
        num_trades: int,
    ) -> DailyPnL:
        """Record daily P&L."""
        with self.get_session() as session:
            # Check if record exists for this date
            existing = session.query(DailyPnL).filter_by(date=date).first()
            if existing:
                existing.starting_equity = starting_equity
                existing.ending_equity = ending_equity
                existing.realized_pnl = realized_pnl
                existing.unrealized_pnl = unrealized_pnl
                existing.total_pnl = realized_pnl + unrealized_pnl
                existing.num_trades = num_trades
                session.commit()
                return existing
            else:
                record = DailyPnL(
                    date=date,
                    starting_equity=starting_equity,
                    ending_equity=ending_equity,
                    realized_pnl=realized_pnl,
                    unrealized_pnl=unrealized_pnl,
                    total_pnl=realized_pnl + unrealized_pnl,
                    num_trades=num_trades,
                )
                session.add(record)
                session.commit()
                session.refresh(record)
                return record

    def get_trades(
        self,
        pair: Optional[tuple[str, str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get trades as DataFrame."""
        with self.get_session() as session:
            query = session.query(Trade)

            if pair:
                query = query.filter(
                    Trade.pair_asset1 == pair[0],
                    Trade.pair_asset2 == pair[1],
                )
            if start_date:
                query = query.filter(Trade.timestamp >= start_date)
            if end_date:
                query = query.filter(Trade.timestamp <= end_date)

            trades = query.order_by(Trade.timestamp.desc()).all()
            return pd.DataFrame([t.to_dict() for t in trades])

    def get_daily_pnl(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get daily P&L records as DataFrame."""
        with self.get_session() as session:
            query = session.query(DailyPnL)

            if start_date:
                query = query.filter(DailyPnL.date >= start_date)
            if end_date:
                query = query.filter(DailyPnL.date <= end_date)

            records = query.order_by(DailyPnL.date.desc()).all()
            return pd.DataFrame([r.to_dict() for r in records])

    def get_latest_cointegration(
        self,
        pair: tuple[str, str],
    ) -> Optional[dict]:
        """Get latest cointegration result for a pair."""
        with self.get_session() as session:
            result = (
                session.query(CointegrationResult)
                .filter(
                    CointegrationResult.asset1 == pair[0],
                    CointegrationResult.asset2 == pair[1],
                )
                .order_by(CointegrationResult.timestamp.desc())
                .first()
            )
            return result.to_dict() if result else None
