# clients/ecommerce_api_client.py
import os
from pathlib import Path
from typing import Dict, Any

from sqlalchemy import create_engine, text

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{(BASE_DIR / 'db' / 'shopassist.db').as_posix()}"


class MockECommerceAPIClient:
    """Interacts with the E-commerce Microservices APIs (CRM, Order DB, Inventory, Helpdesk).

    Despite the name (kept for import compatibility with base_agent.py /
    main_simulation.py / product_recommendation_agent.py), this now hits a
    real database rather than returning hardcoded dicts. Locally that's the
    SQLite dev DB built by db/init_db.py; point DATABASE_URL at the Supabase
    Postgres connection string to switch — no query changes needed.
    """

    def __init__(self, database_url: str = None):
        self.engine = create_engine(database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL))

    def get_order_details(self, customer_id: str, order_id: str) -> Dict[str, Any]:
        print(f"  [ECommerceAPI] Fetching order details for customer {customer_id}, order {order_id}...")
        try:
            order_id_int = int(order_id)
        except (TypeError, ValueError):
            return {"error": "Order not found", "order_id": order_id}

        with self.engine.connect() as conn:
            order_row = conn.execute(
                text("SELECT order_id, customer_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id_int},
            ).mappings().first()

            if not order_row:
                return {"error": "Order not found", "order_id": order_id}

            # customer_id here is the router/session-layer identifier (e.g. "cust_001"),
            # which isn't wired up to the real numeric customers.customer_id yet (open
            # gap flagged separately). Only enforce the match when we're given something
            # that actually looks like a DB customer_id, so it degrades gracefully.
            try:
                if int(customer_id) != order_row["customer_id"]:
                    return {"error": "Order not found", "order_id": order_id}
            except (TypeError, ValueError):
                pass

            item_rows = conn.execute(
                text(
                    """
                    SELECT i.name, oi.quantity
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    WHERE oi.order_id = :order_id
                    """
                ),
                {"order_id": order_id_int},
            ).mappings().all()

        return {
            "order_id": str(order_row["order_id"]),
            "customer_id": str(order_row["customer_id"]),
            "status": order_row["status"].capitalize(),
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            # schema.sql has no ETA column yet; NLG/StructuredOrderSummary already
            # treat this as optional, so report unknown rather than fabricate a date.
            "estimated_delivery": None,
        }

    def get_customer_history(self, customer_id: str) -> Dict[str, Any]:
        print(f"  [ECommerceAPI] Fetching customer history for {customer_id}...")
        try:
            customer_id_int = int(customer_id)
        except (TypeError, ValueError):
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
                {"customer_id": customer_id_int},
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
                {"customer_id": customer_id_int},
            ).scalar()

        return {"last_purchase": last_purchase, "favorite_category": favorite_category}


if __name__ == "__main__":
    client = MockECommerceAPIClient()
    order = client.get_order_details("1", "12345")
    print(f"Sample Order Details: {order}")
    history = client.get_customer_history("1")
    print(f"Sample Customer History: {history}")
