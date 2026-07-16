# services/ecommerce_client.py
import os
from pathlib import Path
from typing import Dict, Any, Optional

from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent.parent
# The db/ folder that used to live in this repo has moved to the sibling
# shopassist-database repo, which is now the shared source of truth for the
# schema (see that repo's docs/database-design.md). Its SQLite build is a
# checked-out sibling directory, not a package of this project.
DEFAULT_DB_URL = (
    f"sqlite:///{(BASE_DIR.parent / 'shopassist-database' / 'sqlite' / 'database' / 'shopassist.db').as_posix()}"
)


class ECommerceAPIClient:
    """Interacts with the E-commerce Microservices APIs (CRM, Order DB, Inventory, Helpdesk).

    Hits a real database via SQLAlchemy rather than returning hardcoded
    dicts. Locally that's the SQLite dev DB built by shopassist-database's
    sqlite/scripts/create_db.py; set DATABASE_URL to shopassist-database's
    Postgres instance (used for the capstone demo) to switch — no query
    changes needed, since both databases share the same schema.
    """

    def __init__(self, database_url: str = None):
        self.engine = create_engine(database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL))

    @staticmethod
    def _looks_like_customer_id(customer_id: Optional[str]) -> bool:
        """customers.customer_id values are always 'cust-<number>' (see
        shopassist-database). Anything else is a router/session-layer
        identifier (e.g. 'cust_001', a bare '1') that was never wired up to
        the real customer namespace (open gap flagged separately) - treat
        it as "no opinion" rather than a hard mismatch, so lookups degrade
        gracefully instead of rejecting everything that isn't a DB id.
        """
        return bool(customer_id) and customer_id.startswith("cust-")

    def get_order_details(self, customer_id: str, order_id: str) -> Dict[str, Any]:
        print(f"  [ECommerceAPI] Fetching order details for customer {customer_id}, order {order_id}...")
        print(f"  [ECommerceAPI] Using database URL: {self.engine.url}")

        with self.engine.connect() as conn:
            order_row = conn.execute(
                text("SELECT order_id, customer_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                return {"error": "Order not found", "order_id": order_id}

            if self._looks_like_customer_id(customer_id) and customer_id != order_row["customer_id"]:
                return {"error": "Order not found", "order_id": order_id}

            item_rows = conn.execute(
                text(
                    """
                    SELECT i.name, oi.quantity
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    WHERE oi.order_id = :order_id
                    """
                ),
                {"order_id": order_id},
            ).mappings().all()

        return {
            "order_id": order_row["order_id"],
            "customer_id": order_row["customer_id"],
            "status": order_row["status"].capitalize(),
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            # schema.sql has no ETA column yet; NLG/StructuredOrderSummary already
            # treat this as optional, so report unknown rather than fabricate a date.
            "estimated_delivery": None,
        }

    def get_customer_name(self, customer_id: str) -> Optional[str]:
        """First name for greeting the customer. Returns None (never a
        placeholder) if customer_id isn't a real DB customer_id or has no
        match — callers should fall back to a generic greeting, not
        fabricate one.
        """
        if not self._looks_like_customer_id(customer_id):
            return None

        with self.engine.connect() as conn:
            return conn.execute(
                text("SELECT first_name FROM customers WHERE customer_id = :customer_id"),
                {"customer_id": customer_id},
            ).scalar()

    def get_customer_history(self, customer_id: str) -> Dict[str, Any]:
        print(f"  [ECommerceAPI] Fetching customer history for {customer_id}...")
        if not self._looks_like_customer_id(customer_id):
            return {"error": "Customer history not found", "customer_id": customer_id}

        with self.engine.connect() as conn:
            last_purchase = conn.execute(
                text(
                    """
                    SELECT i.name
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.customer_id = :customer_id
                    ORDER BY o.placed_at DESC
                    LIMIT 1
                    """
                ),
                {"customer_id": customer_id},
            ).scalar()

            if last_purchase is None:
                return {"error": "Customer history not found", "customer_id": customer_id}

            favorite_category = conn.execute(
                text(
                    """
                    SELECT i.category
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.customer_id = :customer_id
                    GROUP BY i.category
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                    """
                ),
                {"customer_id": customer_id},
            ).scalar()

        return {"last_purchase": last_purchase, "favorite_category": favorite_category}


if __name__ == "__main__":
    client = ECommerceAPIClient()
    order = client.get_order_details("cust-1001", "ord-1001")
    print(f"Sample Order Details: {order}")
    history = client.get_customer_history("cust-1001")
    print(f"Sample Customer History: {history}")
