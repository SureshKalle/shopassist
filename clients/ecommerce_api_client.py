# clients/ecommerce_api_client.py
from typing import Dict, Any

class MockECommerceAPIClient:
    """Simulates interactions with authoritative E-commerce Microservices APIs (CRM, Order DB, Inventory, Helpdesk)."""
    def get_order_details(self, customer_id: str, order_id: str) -> Dict[str, Any]:
        print(f"  [Mock E-commerce API] Fetching order details for customer {customer_id}, order {order_id}...")
        # Simulate API response
        if order_id == "12345":
            return {"order_id": order_id, "customer_id": customer_id, "status": "Shipped", "items": [{"name": "Widget A", "qty": 1}], "estimated_delivery": "2024-08-10"}
        elif order_id == "54321":
            return {"order_id": order_id, "customer_id": customer_id, "status": "Pending", "items": [{"name": "Gadget X", "qty": 1}], "estimated_delivery": None}
        return {"error": "Order not found", "order_id": order_id}

    def get_customer_history(self, customer_id: str) -> Dict[str, Any]:
        print(f"  [Mock E-commerce API] Fetching customer history for {customer_id}...")
        # Simulate API response
        if customer_id == "cust_001":
             return {"last_purchase": "Laptop Pro", "favorite_category": "Electronics"}
        elif customer_id == "cust_002":
             return {"last_purchase": "Headphones X", "favorite_category": "Audio"}
        return {"error": "Customer history not found", "customer_id": customer_id}

# Example of how this client might be used in a real microservice:
if __name__ == "__main__":
    client = MockECommerceAPIClient()
    order = client.get_order_details("cust_001", "12345")
    print(f"Sample Order Details: {order}")
    history = client.get_customer_history("cust_002")
    print(f"Sample Customer History: {history}")