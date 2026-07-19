-- ShopAssist :: PostgreSQL seed data (IISc alumni shop merchandise, INR pricing)
-- Straight port of db/seed_sqlite.sql - identical INSERT statements (plain
-- ANSI SQL, no SQLite-specific syntax) minus the SQLite-only
-- `PRAGMA foreign_keys` line. See that file for the full seed-data
-- rationale (naming conventions, shipping_fee rule, etc.) - not duplicated
-- here.
--
-- Loaded automatically on first container start via docker-entrypoint-
-- initdb.d, after schema_postgres.sql (see docker-compose.yml).

-- ---------------------------------------------------------------------------
-- customers (10)
-- ---------------------------------------------------------------------------
INSERT INTO customers (user_id, first_name, last_name, email, phone, address_line1, city, state, postal_code, country) VALUES
('alum-1001', 'Aarav', 'Sharma', 'aarav.sharma@example.com', '+91-98450-12345', '12 MG Road', 'Bengaluru', 'Karnataka', '560001', 'India'),
('alum-1002', 'Ananya', 'Iyer', 'ananya.iyer@example.com', '+91-90030-22345', '45 Anna Salai', 'Chennai', 'Tamil Nadu', '600002', 'India'),
('alum-1003', 'Rohan', 'Mehta', 'rohan.mehta@example.com', '+91-98200-33345', '7 Marine Drive', 'Mumbai', 'Maharashtra', '400002', 'India'),
('alum-1004', 'Priya', 'Nair', 'priya.nair@example.com', '+91-94470-44345', '23 MG Road', 'Kochi', 'Kerala', '682016', 'India'),
('alum-1005', 'Vikram', 'Reddy', 'vikram.reddy@example.com', '+91-90000-55345', '88 Banjara Hills', 'Hyderabad', 'Telangana', '500034', 'India'),
('alum-1006', 'Sneha', 'Deshpande', 'sneha.deshpande@example.com', '+91-98220-66345', '15 FC Road', 'Pune', 'Maharashtra', '411004', 'India'),
('alum-1007', 'Ishan', 'Punekar', 'ishan.punekar@example.com', '+91-93717-66345', 'Fatima Nagar', 'Pune', 'Maharashtra', '411014', 'India'),
('alum-1008', 'Kavya', 'Rao', 'kavya.rao@example.com', '+91-99000-77345', '221 Connaught Place', 'New Delhi', 'Delhi', '110001', 'India'),
('alum-1009', 'Arjun', 'Bose', 'arjun.bose@example.com', '+91-98300-88345', '5 Park Street', 'Kolkata', 'West Bengal', '700016', 'India'),
('alum-1010', 'Meera', 'Iyengar', 'meera.iyengar@example.com', '+91-90990-99345', '14 CG Road', 'Ahmedabad', 'Gujarat', '380009', 'India');

-- ---------------------------------------------------------------------------
-- items (75)
-- ---------------------------------------------------------------------------
INSERT INTO items (item_id, name, description, category, price, mrp, stock_quantity) VALUES
-- Apparel (18)
('item-1001', 'Crest T-Shirt', 'Cotton crew-neck tee with the institute crest embroidered on the chest', 'Apparel', 499.00, 558.88, 150),
('item-1002', 'Alumni Hoodie', 'Fleece-lined pullover hoodie with ''IISc Alumni'' print on the back', 'Apparel', 1299.00, 1454.88, 80),
('item-1003', 'Polo Shirt', 'Pique cotton polo with embroidered institute crest', 'Apparel', 799.00, 894.88, 120),
('item-1004', 'Baseball Cap', 'Adjustable cotton cap with embroidered logo', 'Apparel', 349.00, 390.88, 200),
('item-1005', 'Convocation Stole', 'Silk-blend stole in institute colours, worn at convocation', 'Apparel', 899.00, 1006.88, 60),
('item-1006', 'Alumni Jacket', 'Windcheater jacket with alumni print on the back', 'Apparel', 1799.00, 2014.88, 45),
('item-1007', 'Zip-Up Sweatshirt', 'Cotton fleece zip-up hoodie with crest patch', 'Apparel', 1199.00, 1342.88, 70),
('item-1008', 'Crest Snapback Cap', 'Flat-brim snapback cap with embroidered crest', 'Apparel', 399.00, 446.88, 90),
('item-1009', 'Alumni Scarf', 'Woven scarf in institute colours', 'Apparel', 599.00, 670.88, 55),
('item-1010', 'Institute Socks (Pack of 3)', 'Cotton crew socks with crest motif, pack of 3', 'Apparel', 299.00, 334.88, 180),
('item-1011', 'Women''s Fit Crest Tee', 'Fitted crew-neck tee with crest print', 'Apparel', 549.00, 614.88, 100),
('item-1012', 'Alumni Track Pants', 'Cotton-blend joggers with side stripe', 'Apparel', 999.00, 1118.88, 65),
('item-1013', 'Crest Muffler', 'Winter muffler in maroon and gold', 'Apparel', 449.00, 502.88, 70),
('item-1014', 'Alumni Rain Jacket', 'Packable waterproof shell jacket', 'Apparel', 1599.00, 1790.88, 40),
('item-1015', 'Institute Bandana', 'Printed cotton bandana with crest motif', 'Apparel', 199.00, 222.88, 120),
('item-1016', 'Convocation Pin Set', 'Enamel pin set for convocation blazers', 'Apparel', 249.00, 278.88, 150),
('item-1017', 'Alumni Beanie', 'Ribbed knit winter cap with crest patch', 'Apparel', 349.00, 390.88, 85),
('item-1018', 'Crest Golf Tee', 'Moisture-wicking polo for sports events', 'Apparel', 849.00, 950.88, 60),
-- Drinkware (10)
('item-1019', 'Ceramic Mug', '320ml ceramic mug printed with the institute crest', 'Drinkware', 299.00, 334.88, 250),
('item-1020', 'Steel Tumbler', 'Double-walled stainless steel tumbler, 500ml', 'Drinkware', 599.00, 670.88, 150),
('item-1021', 'Insulated Water Bottle', '750ml vacuum-insulated bottle, keeps drinks cold for 24h', 'Drinkware', 449.00, 502.88, 180),
('item-1022', 'Travel Coffee Mug', '350ml spill-proof travel mug', 'Drinkware', 549.00, 614.88, 100),
('item-1023', 'Copper Water Bottle', '1L pure copper bottle, Ayurvedic design', 'Drinkware', 699.00, 782.88, 90),
('item-1024', 'Glass Sipper Bottle', '600ml borosilicate glass bottle with silicone sleeve', 'Drinkware', 499.00, 558.88, 110),
('item-1025', 'Crest Beer Mug', '400ml oversized ceramic mug with crest', 'Drinkware', 349.00, 390.88, 130),
('item-1026', 'Insulated Coffee Flask', '500ml flask, keeps beverages hot for 12h', 'Drinkware', 799.00, 894.88, 75),
('item-1027', 'Kids Sipper Bottle', '400ml BPA-free sipper bottle for young alumni', 'Drinkware', 299.00, 334.88, 95),
('item-1028', 'Wine Tumbler Set (2)', 'Stainless steel stemless wine tumblers, set of 2', 'Drinkware', 899.00, 1006.88, 50),
-- Stationery (18)
('item-1029', 'Hardbound Notebook', 'A5 ruled notebook with a debossed institute crest', 'Stationery', 249.00, 278.88, 300),
('item-1030', 'Centenary Coffee Table Book', 'Hardbound pictorial history of the institute', 'Stationery', 1499.00, 1678.88, 40),
('item-1031', 'Desk Diary', 'A5 dated diary with ribbon bookmark', 'Stationery', 399.00, 446.88, 120),
('item-1032', 'Sticky Notes Set', 'Crest-branded sticky note pad set', 'Stationery', 149.00, 166.88, 200),
('item-1033', 'Fountain Pen', 'Institute-branded fountain pen with gift case', 'Stationery', 899.00, 1006.88, 55),
('item-1034', 'Ballpoint Pen Set (3)', 'Crest-engraved metal ballpoint pens, set of 3', 'Stationery', 349.00, 390.88, 140),
('item-1035', 'Notebook & Sleeve Combo', 'Notebook and matching laptop sleeve gift set', 'Stationery', 999.00, 1118.88, 45),
('item-1036', 'Desk Calendar', 'Annual desk calendar with campus photography', 'Stationery', 299.00, 334.88, 160),
('item-1037', 'Sketchbook', 'A4 spiral-bound sketchbook, blank pages', 'Stationery', 349.00, 390.88, 90),
('item-1038', 'Bookmark Set (5)', 'Metal bookmarks with crest cutout, set of 5', 'Stationery', 199.00, 222.88, 170),
('item-1039', 'File Folder Set', 'Set of 3 ring-binder folders with crest', 'Stationery', 449.00, 502.88, 100),
('item-1040', 'Sticky Flag Set', 'Assorted page-marker flags', 'Stationery', 129.00, 144.88, 220),
('item-1041', 'Wall Planner', 'Yearly wall planner with campus map', 'Stationery', 349.00, 390.88, 80),
('item-1042', 'Highlighter Set (5)', 'Pastel highlighter set in crest pouch', 'Stationery', 249.00, 278.88, 150),
('item-1043', 'Correction Tape Set', 'Twin-pack correction tape', 'Stationery', 149.00, 166.88, 190),
('item-1044', 'Institute Postcard Set', 'Campus photography postcard set, pack of 10', 'Stationery', 249.00, 278.88, 130),
('item-1045', 'Greeting Card Set', 'Alumni-themed greeting cards, pack of 6', 'Stationery', 299.00, 334.88, 100),
('item-1046', 'Exam Pad', 'A4 hardboard clipboard exam pad', 'Stationery', 199.00, 222.88, 140),
-- Bags & Accessories (14)
('item-1047', 'Laptop Backpack', 'Padded 15.6-inch laptop backpack with crest', 'Bags & Accessories', 1899.00, 2126.88, 60),
('item-1048', 'Canvas Tote Bag', 'Cotton canvas tote bag with alumni print', 'Bags & Accessories', 399.00, 446.88, 150),
('item-1049', 'Laptop Sleeve', 'Neoprene 14-inch laptop sleeve', 'Bags & Accessories', 599.00, 670.88, 100),
('item-1050', 'Drawstring Gym Bag', 'Polyester drawstring sports bag', 'Bags & Accessories', 349.00, 390.88, 120),
('item-1051', 'Leather Wallet', 'Bifold leather wallet with debossed crest', 'Bags & Accessories', 799.00, 894.88, 70),
('item-1052', 'Crest Keychain', 'Enamel metal keychain with institute crest', 'Bags & Accessories', 149.00, 166.88, 250),
('item-1053', 'Lanyard', 'Woven lanyard with breakaway safety clip', 'Bags & Accessories', 129.00, 144.88, 300),
('item-1054', 'Duffel Bag', 'Weekend duffel bag with shoe compartment', 'Bags & Accessories', 1599.00, 1790.88, 40),
('item-1055', 'Passport Cover', 'Faux-leather passport holder, embossed crest', 'Bags & Accessories', 349.00, 390.88, 110),
('item-1056', 'Card Holder', 'Slim RFID-blocking card holder', 'Bags & Accessories', 299.00, 334.88, 130),
('item-1057', 'Crest Umbrella', '3-fold auto-open umbrella with crest print', 'Bags & Accessories', 599.00, 670.88, 90),
('item-1058', 'Tote Cooler Bag', 'Insulated lunch/cooler tote bag', 'Bags & Accessories', 649.00, 726.88, 75),
('item-1059', 'Sling Bag', 'Crossbody sling bag with crest patch', 'Bags & Accessories', 799.00, 894.88, 65),
('item-1060', 'Luggage Tag Set (2)', 'Leatherette luggage tags with name card, set of 2', 'Bags & Accessories', 249.00, 278.88, 140),
-- Home & Decor (9)
('item-1061', 'Photo Frame', 'Wooden desk photo frame with engraved crest', 'Home & Decor', 399.00, 446.88, 100),
('item-1062', 'Campus Wall Art Print', 'Framed print of the institute''s main building', 'Home & Decor', 1299.00, 1454.88, 40),
('item-1063', 'Desk Organizer', 'Wooden multi-slot desk organizer', 'Home & Decor', 699.00, 782.88, 70),
('item-1064', 'Crest Paperweight', 'Brass paperweight with engraved crest', 'Home & Decor', 499.00, 558.88, 90),
('item-1065', 'Coaster Set (6)', 'Wooden coaster set with campus motifs, set of 6', 'Home & Decor', 449.00, 502.88, 110),
('item-1066', 'Table Clock', 'Wooden desk clock with crest dial', 'Home & Decor', 899.00, 1006.88, 55),
('item-1067', 'Wall Clock', 'Analog wall clock in institute colours', 'Home & Decor', 1099.00, 1230.88, 40),
('item-1068', 'Cushion Cover Set (2)', 'Printed cushion covers with crest motif, set of 2', 'Home & Decor', 599.00, 670.88, 80),
('item-1069', 'Wooden Nameplate', 'Engraved wooden desk nameplate', 'Home & Decor', 549.00, 614.88, 60),
-- Electronics & Gadgets (6)
('item-1070', 'USB Flash Drive 32GB', 'Metal-body pen drive with crest engraving', 'Electronics', 599.00, 670.88, 100),
('item-1071', 'Power Bank 10000mAh', 'Compact power bank with crest print', 'Electronics', 1299.00, 1454.88, 70),
('item-1072', 'Wireless Mouse', 'Ergonomic wireless mouse in institute colours', 'Electronics', 799.00, 894.88, 65),
('item-1073', 'Mobile Stand', 'Foldable aluminium phone/tablet stand', 'Electronics', 349.00, 390.88, 120),
('item-1074', 'Bluetooth Speaker', 'Portable Bluetooth speaker with crest badge', 'Electronics', 1499.00, 1678.88, 50),
('item-1075', 'Laptop Cooling Pad', 'USB-powered laptop cooling pad, dual fan', 'Electronics', 999.00, 1118.88, 45);

-- ---------------------------------------------------------------------------
-- sessions (10)
-- ---------------------------------------------------------------------------
INSERT INTO sessions (session_id, user_id, device_type, status) VALUES
('sess-1001', 'alum-1001', 'web', 'expired'),
('sess-1002', 'alum-1002', 'mobile', 'expired'),
('sess-1003', 'alum-1003', 'web', 'expired'),
('sess-1004', 'alum-1004', 'desktop', 'expired'),
('sess-1005', 'alum-1005', 'mobile', 'expired'),
('sess-1006', 'alum-1006', 'web', 'expired'),
('sess-1007', 'alum-1007', 'mobile', 'expired'),
('sess-1008', 'alum-1008', 'web', 'expired'),
('sess-1009', 'alum-1009', 'desktop', 'expired'),
('sess-1010', 'alum-1010', 'web', 'active');

-- ---------------------------------------------------------------------------
-- orders (25)
-- ---------------------------------------------------------------------------
INSERT INTO orders (order_id, user_id, session_id, status, subtotal, discount, shipping_fee, total_amount, shipping_address, placed_at) VALUES
('ord-1001', 'alum-1001', 'sess-1001', 'delivered',  1197.00,   0.00,  0.00, 1197.00, '12 MG Road, Bengaluru, Karnataka 560001, India', '2026-05-15 10:00:00'),
('ord-1002', 'alum-1002', 'sess-1002', 'delivered',   598.00,   0.00, 49.00,  647.00, '45 Anna Salai, Chennai, Tamil Nadu 600002, India', '2026-05-17 09:10:00'),
('ord-1003', 'alum-1003', 'sess-1003', 'pending',    1499.00,   0.00,  0.00, 1499.00, '7 Marine Drive, Mumbai, Maharashtra 400002, India', '2026-05-20 17:45:00'),
('ord-1004', 'alum-1004', 'sess-1004', 'confirmed',   598.00,   0.00, 49.00,  647.00, '23 MG Road, Kochi, Kerala 682016, India', '2026-05-22 11:05:00'),
('ord-1005', 'alum-1005', 'sess-1005', 'delivered',  1748.00,   0.00,  0.00, 1748.00, '88 Banjara Hills, Hyderabad, Telangana 500034, India', '2026-05-25 08:20:00'),
('ord-1006', 'alum-1001', 'sess-1001', 'shipped',     799.00,   0.00, 49.00,  848.00, '12 MG Road, Bengaluru, Karnataka 560001, India', '2026-05-28 10:00:00'),
('ord-1007', 'alum-1001', 'sess-1001', 'pending',     349.00,   0.00, 49.00,  398.00, '12 MG Road, Bengaluru, Karnataka 560001, India', '2026-06-01 16:15:00'),
('ord-1008', 'alum-1002', 'sess-1002', 'delivered',  1899.00, 100.00,  0.00, 1799.00, '45 Anna Salai, Chennai, Tamil Nadu 600002, India', '2026-06-03 12:00:00'),
('ord-1009', 'alum-1002', 'sess-1002', 'cancelled',  1599.00,   0.00,  0.00, 1599.00, '45 Anna Salai, Chennai, Tamil Nadu 600002, India', '2026-06-05 14:30:00'),
('ord-1010', 'alum-1003', 'sess-1003', 'delivered',   847.00,   0.00, 49.00,  896.00, '7 Marine Drive, Mumbai, Maharashtra 400002, India', '2026-06-07 09:45:00'),
('ord-1011', 'alum-1004', 'sess-1004', 'delivered',   697.00,   0.00, 49.00,  746.00, '23 MG Road, Kochi, Kerala 682016, India', '2026-06-10 13:20:00'),
('ord-1012', 'alum-1005', 'sess-1005', 'shipped',    1299.00,   0.00,  0.00, 1299.00, '88 Banjara Hills, Hyderabad, Telangana 500034, India', '2026-06-12 15:00:00'),
('ord-1013', 'alum-1005', 'sess-1005', 'delivered',  1499.00, 150.00,  0.00, 1349.00, '88 Banjara Hills, Hyderabad, Telangana 500034, India', '2026-06-14 11:40:00'),
('ord-1014', 'alum-1006', 'sess-1006', 'delivered',  1799.00,   0.00,  0.00, 1799.00, '15 FC Road, Pune, Maharashtra 411004, India', '2026-06-17 10:10:00'),
('ord-1015', 'alum-1006', 'sess-1006', 'returned',   1599.00,   0.00,  0.00, 1599.00, '15 FC Road, Pune, Maharashtra 411004, India', '2026-06-19 09:30:00'),
('ord-1016', 'alum-1007', 'sess-1007', 'delivered',   697.00,   0.00, 49.00,  746.00, 'Fatima Nagar, Pune, Maharashtra 411014, India', '2026-06-22 14:00:00'),
('ord-1017', 'alum-1007', 'sess-1007', 'pending',     599.00,   0.00, 49.00,  648.00, 'Fatima Nagar, Pune, Maharashtra 411014, India', '2026-06-25 16:45:00'),
('ord-1018', 'alum-1007', 'sess-1007', 'confirmed',   599.00,   0.00, 49.00,  648.00, 'Fatima Nagar, Pune, Maharashtra 411014, India', '2026-06-28 12:15:00'),
('ord-1019', 'alum-1008', 'sess-1008', 'delivered',  1299.00,   0.00,  0.00, 1299.00, '221 Connaught Place, New Delhi, Delhi 110001, India', '2026-07-01 10:00:00'),
('ord-1020', 'alum-1008', 'sess-1008', 'shipped',    1398.00,   0.00,  0.00, 1398.00, '221 Connaught Place, New Delhi, Delhi 110001, India', '2026-07-03 11:30:00'),
('ord-1021', 'alum-1009', 'sess-1009', 'delivered',  2498.00, 200.00,  0.00, 2298.00, '5 Park Street, Kolkata, West Bengal 700016, India', '2026-07-05 09:00:00'),
('ord-1022', 'alum-1009', 'sess-1009', 'pending',     999.00,   0.00,  0.00,  999.00, '5 Park Street, Kolkata, West Bengal 700016, India', '2026-07-08 15:20:00'),
('ord-1023', 'alum-1009', 'sess-1009', 'confirmed',  1398.00,   0.00,  0.00, 1398.00, '5 Park Street, Kolkata, West Bengal 700016, India', '2026-07-10 13:00:00'),
('ord-1024', 'alum-1010', 'sess-1010', 'shipped',    1648.00,   0.00,  0.00, 1648.00, '14 CG Road, Ahmedabad, Gujarat 380009, India', '2026-07-13 10:45:00'),
('ord-1025', 'alum-1010', 'sess-1010', 'pending',    1947.00, 100.00,  0.00, 1847.00, '14 CG Road, Ahmedabad, Gujarat 380009, India', '2026-07-16 17:00:00');

-- ---------------------------------------------------------------------------
-- order_items (38)
-- ---------------------------------------------------------------------------
INSERT INTO order_items (order_id, item_id, quantity, unit_price, line_total) VALUES
('ord-1001', 'item-1001', 1,  499.00,  499.00),
('ord-1001', 'item-1004', 2,  349.00,  698.00),
('ord-1002', 'item-1010', 1,  299.00,  299.00),
('ord-1002', 'item-1019', 1,  299.00,  299.00),
('ord-1003', 'item-1030', 1, 1499.00, 1499.00),
('ord-1004', 'item-1019', 2,  299.00,  598.00),
('ord-1005', 'item-1002', 1, 1299.00, 1299.00),
('ord-1005', 'item-1021', 1,  449.00,  449.00),
('ord-1006', 'item-1003', 1,  799.00,  799.00),
('ord-1007', 'item-1004', 1,  349.00,  349.00),
('ord-1008', 'item-1047', 1, 1899.00, 1899.00),
('ord-1009', 'item-1054', 1, 1599.00, 1599.00),
('ord-1010', 'item-1029', 2,  249.00,  498.00),
('ord-1010', 'item-1034', 1,  349.00,  349.00),
('ord-1011', 'item-1048', 1,  399.00,  399.00),
('ord-1011', 'item-1052', 2,  149.00,  298.00),
('ord-1012', 'item-1071', 1, 1299.00, 1299.00),
('ord-1013', 'item-1074', 1, 1499.00, 1499.00),
('ord-1014', 'item-1006', 1, 1799.00, 1799.00),
('ord-1015', 'item-1014', 1, 1599.00, 1599.00),
('ord-1016', 'item-1019', 1,  299.00,  299.00),
('ord-1016', 'item-1029', 1,  249.00,  249.00),
('ord-1016', 'item-1052', 1,  149.00,  149.00),
('ord-1017', 'item-1057', 1,  599.00,  599.00),
('ord-1018', 'item-1009', 1,  599.00,  599.00),
('ord-1019', 'item-1062', 1, 1299.00, 1299.00),
('ord-1020', 'item-1066', 1,  899.00,  899.00),
('ord-1020', 'item-1064', 1,  499.00,  499.00),
('ord-1021', 'item-1047', 1, 1899.00, 1899.00),
('ord-1021', 'item-1049', 1,  599.00,  599.00),
('ord-1022', 'item-1075', 1,  999.00,  999.00),
('ord-1023', 'item-1072', 1,  799.00,  799.00),
('ord-1023', 'item-1070', 1,  599.00,  599.00),
('ord-1024', 'item-1002', 1, 1299.00, 1299.00),
('ord-1024', 'item-1004', 1,  349.00,  349.00),
('ord-1025', 'item-1030', 1, 1499.00, 1499.00),
('ord-1025', 'item-1044', 1,  249.00,  249.00),
('ord-1025', 'item-1038', 1,  199.00,  199.00);

-- ---------------------------------------------------------------------------
-- item_reviews (10) - user_id is a real FK to customers, one of the 10
-- seeded above (schema_postgres.sql's file header explains why); reviewers
-- here happen to be reviewing items they've actually bought in the
-- order_items seed data above, though the schema doesn't require that.
-- ---------------------------------------------------------------------------
INSERT INTO item_reviews (review_id, item_id, user_id, review_title, review_content) VALUES
('rev-1001', 'item-1001', 'alum-1001', 'Good quality', 'Fabric feels durable and the print hasn''t faded after several washes.'),
('rev-1002', 'item-1004', 'alum-1001', 'Fits well', 'Adjustable strap makes it comfortable for everyday wear.'),
('rev-1003', 'item-1002', 'alum-1005', 'Warm and cozy', 'Great for Bengaluru winters, the fleece lining is soft.'),
('rev-1004', 'item-1019', 'alum-1002', 'Nice mug', 'Crest print looks sharp, handle is sturdy.'),
('rev-1005', 'item-1019', 'alum-1004', 'As described', 'Good size for coffee, arrived well packaged.'),
('rev-1006', 'item-1029', 'alum-1003', 'Good paper quality', 'Ruled pages are smooth, no bleed-through with gel pens.'),
('rev-1007', 'item-1047', 'alum-1002', 'Sturdy backpack', 'Fits a 15-inch laptop easily, padding is solid.'),
('rev-1008', 'item-1047', 'alum-1009', 'Value for money', 'Zippers feel a bit light but overall happy with the purchase.'),
('rev-1009', 'item-1070', 'alum-1009', 'Fast transfer speeds', 'Metal body feels premium and transfer is quick.'),
('rev-1010', 'item-1074', 'alum-1005', 'Decent sound', 'Bass is a bit weak but fine for casual listening.');
