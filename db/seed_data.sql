-- Sample data for local dev (db/shopassist.db).
-- order_id 12345 / 54321 are seeded explicitly so they match the values already
-- referenced in main_simulation.py and the old MockECommerceAPIClient responses.

INSERT INTO customers (customer_id, first_name, last_name, email, city, country) VALUES
    (1, 'John',  'Doe',   'john.doe@example.com',   'Bengaluru', 'India'),
    (2, 'Jane',  'Smith', 'jane.smith@example.com', 'Mumbai',    'India'),
    (3, 'Alex',  'Chen',  'alex.chen@example.com',  'Pune',      'India'),
    (4, 'Priya', 'Nair',  'priya.nair@example.com', 'Chennai',   'India');

INSERT INTO items (item_id, sku, name, description, category, price, mrp, stock_quantity) VALUES
    (1, 'LAP-001',  'Gaming Laptop Pro',                   'High-performance gaming laptop, i7/16GB/1TB SSD.', 'Electronics', 1200.00, 1350.00, 25),
    (2, 'HEAD-002', 'Premium Noise-Cancelling Headphones', 'Immersive audio, 20-hour battery life.',           'Audio',       250.00,  280.00,  60),
    (3, 'WID-A',    'Widget A',                            'Sample accessory item.',                           'Accessories', 49.99,   59.99,   100),
    (4, 'GAD-X',    'Gadget X',                            'Sample electronics item.',                         'Electronics', 89.99,   99.99,   40);

INSERT INTO sessions (session_id, customer_id, device_type, status) VALUES
    ('sess-demo-0001', 1, 'web', 'active');

INSERT INTO orders (order_id, customer_id, session_id, status, subtotal, discount, shipping_fee, total_amount, shipping_address) VALUES
    (12345, 1, 'sess-demo-0001', 'shipped',   49.99, 0, 5.00, 54.99, '221B Baker Street, Bengaluru'),
    (54321, 1, NULL,             'pending',   89.99, 0, 0.00, 89.99, '221B Baker Street, Bengaluru'),
    (3,     2, NULL,             'delivered', 1200.00, 0, 0.00, 1200.00, '45 MG Road, Mumbai');

INSERT INTO order_items (order_id, item_id, quantity, unit_price, line_total) VALUES
    (12345, 3, 1, 49.99,   49.99),
    (54321, 4, 1, 89.99,   89.99),
    (3,     1, 1, 1200.00, 1200.00);
