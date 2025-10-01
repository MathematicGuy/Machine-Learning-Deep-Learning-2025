DROP DATABASE IF EXISTS sql_store;
CREATE DATABASE sql_store;
USE sql_store;

-- ########################## --
DROP TABLE IF EXISTS products;
CREATE TABLE products (
        product_id INT PRIMARY KEY NOT NULL  AUTO_INCREMENT,
        name VARCHAR(50) NOT NULL,
        quantity_in_stock INT NOT NULL,
        unit_price DECIMAL(4,2) NOT NULL
);
-- ########################## --
DROP TABLE IF EXISTS customers;
CREATE TABLE customers (
        customer_id INT NOT NULL AUTO_INCREMENT,
        first_name VARCHAR(50) NOT NULL,
        last_name VARCHAR(50) NOT NULL,
        birth_date DATE DEFAULT NULL,
        phone VARCHAR(50) DEFAULT NULL,
        address VARCHAR(50) NOT NULL,
        city VARCHAR(50) NOT NULL,
        state CHAR(2) NOT NULL,
        points INT NOT NULL DEFAULT '0',
        PRIMARY KEY (customer_id)
);
-- ########################## --
DROP TABLE IF EXISTS shippers;
CREATE TABLE shippers (
        shipper_id INT NOT NULL AUTO_INCREMENT,
        name VARCHAR(50) NOT NULL,
        PRIMARY KEY (shipper_id)
);
-- ########################## --
DROP TABLE IF EXISTS order_statuses;
CREATE TABLE order_statuses (
        order_status_id INT NOT NULL,
        name VARCHAR(50) NOT NULL,
        PRIMARY KEY (order_status_id)
);
-- ########################## --
DROP TABLE IF EXISTS orders;
CREATE TABLE orders (
        order_id INT PRIMARY KEY NOT NULL AUTO_INCREMENT,
        customer_id INT NOT NULL,
        order_date DATE NOT NULL,
        status INT NOT NULL DEFAULT '1',
        comments VARCHAR(2000) DEFAULT NULL,
        shipped_date DATE DEFAULT NULL,
        shipper_id INT DEFAULT NULL,
        CONSTRAINT fk_orders_customers FOREIGN KEY (customer_id) 
			REFERENCES customers(customer_id) ON UPDATE CASCADE,
        CONSTRAINT fk_orders_order_statuses FOREIGN KEY (status) 
			REFERENCES order_statuses(order_status_id) ON UPDATE CASCADE,
        CONSTRAINT fk_orders_shippers FOREIGN KEY (shipper_id) 
			REFERENCES shippers (shipper_id) ON UPDATE CASCADE
);
-- ########################## --
DROP TABLE IF EXISTS order_items;
CREATE TABLE order_items (
        order_id INT NOT NULL AUTO_INCREMENT,
        product_id INT NOT NULL,
        quantity INT NOT NULL,
        unit_price DECIMAL(4,2) NOT NULL,
        PRIMARY KEY (order_id, product_id),
        CONSTRAINT fk_order_items_orders FOREIGN KEY (order_id)
			REFERENCES orders(order_id) ON UPDATE CASCADE,
        CONSTRAINT fk_order_items_products FOREIGN KEY (product_id)
			REFERENCES products (product_id) ON UPDATE CASCADE
);
-- ########################## --




