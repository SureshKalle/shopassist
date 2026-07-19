# clients/ecommerce_api_client.py
import logging
import os
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DB_URL = f"sqlite:///{(BASE_DIR / 'db' / 'shopassist.db').as_posix()}"

# orders.status CHECK constraint (db/schema_sqlite.sql) also allows
# 'delivered', 'cancelled', 'returned' - those are terminal, not cancellable.
_CANCELLABLE_STATUSES = {"pending", "confirmed", "shipped"}


class EcommerceClient:
    """Data-access layer for the orders/customers/items DB - the one place
    every other module goes through instead of touching SQL or DATABASE_URL
    directly. Locally that's the SQLite dev DB built by db/init_db.py; point
    DATABASE_URL at a real PostgreSQL instance to switch, once that
    database's schema is aligned with this one (see db/README.md).
    Everything here keys on user_id - shopassist's single identifier, sent
    by shopassist-client at login - end to end.
    """

    def __init__(self, database_url: str = None):
        self.engine = create_engine(database_url or os.environ.get("DATABASE_URL", DEFAULT_DB_URL))
        logger.info("EcommerceClient initialised (engine=%s)", self.engine.url)

    def is_reachable(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.debug("Database reachability check: OK")
            return True
        except Exception:
            logger.warning("Database not reachable", exc_info=True)
            return False

    def get_order_details(self, user_id: str, order_id: str) -> dict[str, Any]:
        logger.info("get_order_details: user_id=%s order_id=%s", user_id, order_id)
        logger.debug("Using database URL: %s", self.engine.url)

        with self.engine.connect() as conn:
            order_row = conn.execute(
                text("SELECT order_id, user_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                logger.warning("get_order_details: order_id=%s not found in orders table", order_id)
                return {"error": "Order not found", "order_id": order_id}

            logger.debug("get_order_details: row found - %s", dict(order_row))

            # Ownership check: a caller can only see their own orders. Returns
            # the same "not found" error as a missing order_id rather than a
            # distinct "forbidden" - that keeps this indistinguishable from a
            # typo'd order_id to the caller, instead of confirming that a
            # given order_id exists but belongs to someone else.
            if user_id != order_row["user_id"]:
                logger.warning(
                    "get_order_details: ownership mismatch - user_id=%s requested order_id=%s owned by %s",
                    user_id, order_id, order_row["user_id"],
                )
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
            logger.debug("get_order_details: %d line item(s) for order_id=%s", len(item_rows), order_id)

        logger.info("get_order_details: order_id=%s status=%s", order_row["order_id"], order_row["status"])
        return {
            "order_id": order_row["order_id"],
            "user_id": order_row["user_id"],
            "status": order_row["status"].capitalize(),
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            # schema_sqlite.sql has no ETA column yet; NLG/StructuredOrderSummary
            # already treat this as optional, so report unknown rather than
            # fabricate a date.
            "estimated_delivery": None,
        }

    def cancel_order(self, user_id: str, order_id: str) -> dict[str, Any]:
        """Cancel an order in place. Returns the same shape as
        get_order_details() (order_id/user_id/status/items/estimated_delivery)
        so callers can feed either tool's result through the same
        diagnose/summarize pipeline without branching on which one ran - see
        services/agents/order_tracking_agent.py's tool registry.
        """
        logger.info("cancel_order: user_id=%s order_id=%s", user_id, order_id)

        with self.engine.begin() as conn:
            order_row = conn.execute(
                text("SELECT order_id, user_id, status, total_amount FROM orders WHERE order_id = :order_id"),
                {"order_id": order_id},
            ).mappings().first()

            if not order_row:
                logger.warning("cancel_order: order_id=%s not found in orders table", order_id)
                return {"error": "Order not found", "order_id": order_id}

            if user_id != order_row["user_id"]:
                logger.warning(
                    "cancel_order: ownership mismatch - user_id=%s requested order_id=%s owned by %s",
                    user_id, order_id, order_row["user_id"],
                )
                return {"error": "Order not found", "order_id": order_id}

            current_status = order_row["status"].lower()
            if current_status not in _CANCELLABLE_STATUSES:
                logger.info("cancel_order: order_id=%s status=%s is not cancellable", order_id, current_status)
                return {"error": f"Order is already {current_status} and can no longer be cancelled", "order_id": order_id}

            conn.execute(
                text("UPDATE orders SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP WHERE order_id = :order_id"),
                {"order_id": order_id},
            )

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

        logger.info("cancel_order: order_id=%s cancelled (was %s)", order_id, current_status)
        return {
            "order_id": order_row["order_id"],
            "user_id": order_row["user_id"],
            "status": "Cancelled",
            "items": [{"name": row["name"], "qty": row["quantity"]} for row in item_rows],
            "estimated_delivery": None,
        }

    def get_customer_history(self, user_id: str) -> dict[str, Any]:
        logger.info("get_customer_history: user_id=%s", user_id)

        with self.engine.connect() as conn:
            last_purchase = conn.execute(
                text(
                    """
                    SELECT i.name
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.user_id = :user_id
                    ORDER BY o.placed_at DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id},
            ).scalar()

            if last_purchase is None:
                logger.warning("get_customer_history: no order history for user_id=%s", user_id)
                return {"error": "Customer history not found", "user_id": user_id}

            favorite_category = conn.execute(
                text(
                    """
                    SELECT i.category
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.user_id = :user_id
                    GROUP BY i.category
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                    """
                ),
                {"user_id": user_id},
            ).scalar()

        logger.debug(
            "get_customer_history: user_id=%s last_purchase=%s favorite_category=%s",
            user_id, last_purchase, favorite_category,
        )
        return {"last_purchase": last_purchase, "favorite_category": favorite_category}

    def get_customer(self, user_id: str) -> dict[str, Any] | None:
        logger.info("get_customer: user_id=%s", user_id)
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT user_id, first_name, last_name, email, phone, city, state, country
                    FROM customers WHERE user_id = :user_id
                    """
                ),
                {"user_id": user_id},
            ).mappings().first()
        if not row:
            logger.warning("get_customer: user_id=%s not found", user_id)
            return None
        return dict(row)

    def get_item(self, item_id: str) -> dict[str, Any] | None:
        logger.info("get_item: item_id=%s", item_id)
        with self.engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT item_id, name, description, category, price, mrp, stock_quantity
                    FROM items WHERE item_id = :item_id
                    """
                ),
                {"item_id": item_id},
            ).mappings().first()
        if not row:
            logger.warning("get_item: item_id=%s not found", item_id)
            return None
        return dict(row)

    def search_items(self, category: str | None = None, keyword: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        logger.info("search_items: category=%s keyword=%s limit=%d", category, keyword, limit)
        clauses = ["is_active"]
        params: dict[str, Any] = {"limit": limit}
        if category:
            clauses.append("category = :category")
            params["category"] = category
        if keyword:
            clauses.append("(name LIKE :keyword OR description LIKE :keyword)")
            params["keyword"] = f"%{keyword}%"
        where = " AND ".join(clauses)

        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT item_id, name, description, category, price, mrp, stock_quantity
                    FROM items WHERE {where} ORDER BY name LIMIT :limit
                    """
                ),
                params,
            ).mappings().all()
        logger.debug("search_items: %d result(s)", len(rows))
        return [dict(row) for row in rows]

    def list_orders_for_customer(self, user_id: str, limit: int = 10) -> list[dict[str, Any]]:
        logger.info("list_orders_for_customer: user_id=%s limit=%d", user_id, limit)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT order_id, status, total_amount, placed_at
                    FROM orders WHERE user_id = :user_id
                    ORDER BY placed_at DESC LIMIT :limit
                    """
                ),
                {"user_id": user_id, "limit": limit},
            ).mappings().all()
        logger.debug("list_orders_for_customer: %d order(s) for user_id=%s", len(rows), user_id)
        return [dict(row) for row in rows]

    def get_popular_category(self, limit: int = 1) -> list[dict[str, Any]]:
        """Categories ranked by total units sold, most popular first.

        Excludes cancelled/returned orders so a cancellation doesn't still
        count toward a category's popularity.
        """
        logger.info("get_popular_category: limit=%d", limit)
        with self.engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT i.category, SUM(oi.quantity) AS units_sold
                    FROM order_items oi
                    JOIN items i ON i.item_id = oi.item_id
                    JOIN orders o ON o.order_id = oi.order_id
                    WHERE o.status NOT IN ('cancelled', 'returned') AND i.category IS NOT NULL
                    GROUP BY i.category
                    ORDER BY units_sold DESC
                    LIMIT :limit
                    """
                ),
                {"limit": limit},
            ).mappings().all()
        logger.debug("get_popular_category: %d categor(y/ies)", len(rows))
        return [dict(row) for row in rows]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    client = EcommerceClient()
    order = client.get_order_details("alum-1001", "ord-1001")
    print(f"Sample Order Details: {order}")
    history = client.get_customer_history("alum-1001")
    print(f"Sample Customer History: {history}")
    print(f"Reachable: {client.is_reachable()}")
