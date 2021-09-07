from uuid import uuid1


# class for reading file
class Records:
    # customers and products list
    customers = []
    products = []
    combdata = []
    custinst = {}
    cust_name = {}
    cust_id_name = {}

    # Function to read customers.txt file
    def readCustomers(self):
        fileName = "customers.txt"
        # For handling file open error
        try:
            # Reading file and adding customer info in customers list
            f = open(fileName, "r")
            data = f.read()
            lines = data.split('\n')
            for line in lines:
                items = line.split(', ')
                # Using dictionary for easy accessing of customers info
                obj = {
                    'id': items[0],
                    'name': items[1],
                    'customerType': items[2],
                    'discountRate': items[3],
                    'total': items[4],
                }
                # Adding new item in the list
                self.customers.append(obj)
            for data in self.customers:
                self.custinst.setdefault(data["name"], data["customerType"])
                self.cust_name.setdefault(data["id"], data["name"])
                self.cust_id_name.setdefault(data["id"], data["customerType"])
        except IOError:
            # if nor able to open file then print error message and exit
            print("Error: Unable to open the customers.txt file.")
            exit()
        except IndexError:
            # if file not found then print error message and exit
            print("Error: customers.txt is empty or list index out of range")
            exit()

    # Function to read products.txt file
    def readProducts(self):
        fileName = "products.txt"
        # For handling file open error
        try:
            # Reading file and adding product info in products list
            f = open(fileName, "r")
            data = f.read()
            lines = data.split('\n')
            for line in lines:
                items = line.split(', ')
                if items[0].startswith('P'):
                    # Using dictionary for easy accessing of products info
                    obj = {
                        'id': items[0],
                        'name': items[1],
                        'pricePerUnit': items[2],
                        'instockQuantity': items[3],
                    }
                    # Adding new item in the list
                    self.products.append(obj)
                else:
                    # Using dictionary for easy accessing of products info
                    val = {
                        'comb_id': items[0],
                        'combo_name': items[1],
                        'prd_list': items[2:-1],
                        'instockQuantity': items[-1],
                    }
                    # Adding new item in the list
                    self.combdata.append(val)
        except IOError:
            # If not able to open the file then print error message and exit
            print("Error: Unable to open the products.txt file.")
            exit()
        except IndexError:
            # If file not found then print error message and exit
            print("Error: products.txt is empty or list index out of range")
            exit()

    # Function to update stock after order is placed
    def updateStk(self, product_name, quantity):
        # Stock updated for product
        for data in Records.products:
            if product_name == data["name"]:
                data["instockQuantity"] = str(int(data["instockQuantity"]) - int(quantity))
        # Stock updated for combo
        for data in Records.combdata:
            if product_name == data["combo_name"]:
                data["instockQuantity"] = str(int(data["instockQuantity"]) - int(quantity))

    # Function for searching a customer
    def findCustomer(self, val):
        # Looping in the customers list
        for customer in self.customers:
            # comparing entered value with both name and id
            # and returning corresponding item if they are matched
            if customer['name'] == val or customer['id'] == val:
                return customer

        return None

    # Function for searching a product
    def findProduct(self, val):
        try:
            # Open the products file
            with open('products.txt', "r") as a_file:
                for line in a_file:
                    stripped_line = line.strip()
                    id = stripped_line.split(",")[0]
                    name = stripped_line.split(",")[1].lstrip()
                    if id == val or name == val:
                        return print(stripped_line + '\n')
        except IOError:
            # If file not found then print error message and exit
            print("Error: Unable to open the products.txt file.")
            exit()

    # Function for printing all the customers' details
    def listCustomers(self):
        print('\n************ Customers list ***************')
        for customer in self.customers:
            print(customer['id'] + ', ' + customer['name'] + ', ' + customer['customerType'] + ', ' + str(
                customer['discountRate']) + ', ' + str(customer['total']))
        print()

    # Function for printing all the products details in required format
    def listProducts(self):
        print('\n************ Products list ***************')
        for product in self.products:
            print(product['id'] + ', ' + product['name'] + ', ' + str(product['pricePerUnit']) + ', ' + str(
                product['instockQuantity']))
        for product in self.combdata:
            print(product['comb_id'] + ', ' + product['combo_name'] + ', ' + str(product['prd_list']) + ', ' + str(
                product['instockQuantity']))

        print()


# Class for Customer

class Customer:
    def __init__(self, ID, Name):
        self.ID = ID
        self.Name = Name

    def getID(self):
        return self.ID

    def getName(self):
        return self.Name

    def get_discount(self, price):
        pass


# Class for RetailCustomer

class RetailCustomer(Customer):
    # Defining discount rate of 10 % for retail customer
    rate_of_discount = 0.10

    def __init__(self, ID, Name):
        super().__init__(ID, Name)

    def getdiscount(self):
        return self.rate_of_discount

    def setrate(self, value):
        self.rate_of_discount = value

    def get_discount(self, price):
        return price * self.rate_of_discount

    @property
    def displayCustomer(self):
        return f'Retail customer details : {self.getID()} - {self.getName()} - {self.getdiscount()}'


# Class for WholesaleCustomer
class WholesaleCustomer:
    # Defining threshold and discount range for wholesale customers
    lower_discount = .10
    upper_discount = lower_discount + 0.05
    threshold = 1000

    def __init__(self):
        self.lower_discount = .10
        self.threshold = 1000

    def get_lower_discount(self):
        return self.lower_discount

    def get_upper_discount(self):
        return self.upper_discount

    def get_threshold(self):
        return self.threshold

    def set_threshold(self, value):
        self.threshold = value

    # Check if price is below or above threshold, choose discount rate accordingly
    def get_discount(self, price):
        if price <= self.threshold:
            return price * self.lower_discount
        elif price >= self.threshold:
            return price * self.upper_discount

    def discount_rate(self):
        return f'Lower and Upper discount rates for wholesale customer are: {self.get_lower_discount} {self.get_upper_discount}'

    def setrate(self, lower_value, upper_value):
        if self.lower_discount != .10:
            self.lower_discount = lower_value
            return self.lower_discount
        elif self.upper_discount != .15:
            self.upper_discount = upper_value
            return self.upper_discount
        else:
            return f'Lower and Upper discount rates for wholesale customer are: {self.get_lower_discount} {self.get_upper_discount}'

    # Function to display wholesale customer details
    def displayCustomer(self):
        return f'Wholesales customer details : {self.getID()} - {self.getName()} '


# Class for Product
class Product(Records):
    ID = []
    Name = []
    Price = {}
    Stock = {}
    prd_stock = {}
    prd_price = {}
    p_price = {}

    def prod_id(self):
        for item in self.products:
            self.ID.append(item['id'])
        for item in self.combdata:
            self.ID.append(item['comb_id'])

        return self.ID

    def prod_name(self):
        for item in self.products:
            # if item not in self.Name:
            self.Name.append(item['name'])
        for item in self.combdata:
            # if item not in self.Name:
            self.Name.append(item['combo_name'])

        return self.Name

    def prd_id_price(self):
        x = []
        y = []
        for item in self.products:
            if item['id'].startswith('P'):
                x.append(item['id'])
                y.append(float(item['pricePerUnit']))
        self.prd_price = dict(zip(x, y))
        return self.prd_price

    def prod_price(self):
        x = []
        y = []
        for item in self.products:
            x.append(item['name'])
            y.append(float(item['pricePerUnit']))
        for item in self.combdata:
            a = []
            b = []
            for item in self.products:
                if item['id'].startswith('P'):
                    a.append(item['id'])
                    b.append(float(item['pricePerUnit']))
            self.prd_price = dict(zip(a, b))
            for item in self.combdata:
                x.append(item['combo_name'])
            price = 0
            for item in self.combdata:
                z = item['prd_list']
                for item in z:
                    price = self.prd_price[item] + price
            y.append(float(price * 0.9))

        self.Price = dict(zip(x, y))

        return self.Price

    def pd_price(self):
        x = []
        y = []
        for item in self.products:
            x.append(item['id'])
            y.append(float(item['pricePerUnit']))
        for item in self.combdata:
            a = []
            b = []
            for item in self.products:
                if item['id'].startswith('P'):
                    a.append(item['id'])
                    b.append(float(item['pricePerUnit']))
            self.prd_price = dict(zip(a, b))
            for item in self.combdata:
                x.append(item['comb_id'])
            price = 0
            for item in self.combdata:
                z = item['prd_list']
                for item in z:
                    price = self.prd_price[item] + price
            y.append(float(price * 0.9))

        self.p_price = dict(zip(x, y))

        return self.p_price

    def prod_stock(self):
        x = []
        y = []
        for item in self.products:
            x.append(item['name'])
            y.append(int(item['instockQuantity']))
        for item in self.combdata:
            x.append(item['combo_name'])
            y.append(int(item['instockQuantity']))

        self.Stock = dict(zip(x, y))
        return self.Stock

    def prod_id_stock(self):
        x = []
        y = []
        for item in self.products:
            x.append(item['id'])
            y.append(int(item['instockQuantity']))
        for item in self.combdata:
            x.append(item['comb_id'])
            y.append(int(item['instockQuantity']))

        self.Stock = dict(zip(x, y))
        return self.Stock

    # Function to replenish stock
    def replenish(self):
        # Define a variable to take user input so that it can be used to replace stock of all products with user entered
        # value.
        x = int(input("Enter quantity to replenish the stock: \n"))
        # Using for loop  to iterate through all items of products.
        for d in self.products:
            # If statement to check if existing stock of any individiual product is less than entered quantity
            if int(d["instockQuantity"]) < x:
                # Replacing stock of matched product with user input
                d["instockQuantity"] = x + int(d["instockQuantity"])
        # If statement to check if existing stock of any combo product is less than entered quantity
        for d in self.combdata:
            if int(d["instockQuantity"]) < x:
                # Replacing stock of matched combo product swith user input
                d["instockQuantity"] = x + int(d["instockQuantity"])
        for k, v in self.Stock.items():
            if self.Stock[k] < x:
                # Updating product stock with user entered value
                self.Stock[k] = x + int(self.Stock[k])
        # Displaying message on the screen
        print("Stock Updated!!")
        print('\n************ Products list ***************')
        for item in self.products:
            print(item['name'] + " : " + str(item['instockQuantity']))
        for item in self.combdata:
            print(item['combo_name'] + " : " + str(item['instockQuantity']))
        print()


# Class 'Product'
class combo(Product):
    combo_price = {}
    prod_prc = {}

    # Function to get price of the product on the basis of ID
    def getprice(self):
        x = []
        y = []
        for item in self.products:
            if item['id'].startswith('P'):
                x.append(item['id'])
                y.append(float(item['pricePerUnit']))
        self.prod_prc = dict(zip(x, y))
        name = []
        Total_price = []
        for item in self.combdata:
            name.append(item['combo_name'])
        price = 0
        for item in self.combdata:
            x = item['prd_list']
            for item in x:
                price = self.prod_prc[item] + price
        Total_price.append(float(price * 0.9))
        self.combo_price = dict(zip(name, Total_price))

        return self.combo_price


# Class 'Order(combo)'
class Order(combo, WholesaleCustomer):
    grocery_history = []
    order_value = {}
    special_customer = []
    special_discount = {}

    # Function to register new customer in the system
    def usercreate(self):
        print()
        # Accept user entered customer name
        cust_name = input("Enter your name \n")
        # Check if customer already exists or not
        if cust_name not in self.cust_name.values():
            # Give choice to customer to register or not for non-existing cutomer
            choice = str(input("User doesnt exit\nDo you wish to create the user Y/N ?\n"))
            # Ask type of customer if customer wish to get registered
            if choice == 'Y':
                customerType = input("would you like to register yourself?\n"
                                     "Type 'R' to register as RetailCustomer or 'W' to register as WholesaleCustomer:\n")

                ID = int(uuid1())

                # Generate customer record
                val = {"id": int(ID), "name": cust_name, "customerType": customerType,
                       "discountRate": str(0), "total": str(0)}
                Records.customers.append(val)
                Records.cust_name.setdefault(ID, cust_name)
                Records.custinst.setdefault(cust_name, customerType)
                Records.cust_id_name.setdefault(ID, customerType)
                self.order_place(cust_name)
            else:
                self.order_place(cust_name)
        else:
            self.order_place(cust_name)

    # Function to place an order
    def order_place(self, cust_name):

        # variable used to check if the while loop condition is met
        stop = False

        # Using while loop to take customer order until he/she is finished with his/her order
        while not stop:

            # Using while loop until user enters a valid product
            while True:
                # Accept input of the name of the grocery item purchased
                product_name = input("Product name \n")
                # Using if-else loop to check user input with our pre-defined product list
                if product_name not in self.Name:
                    # Print error message on screen for product not present in inventory
                    print("Invalid Product!!Please enter product present in our inventory")
                    # Print products available in inventory
                    print("Below are the products available")
                    for item in self.Name:
                        print(item, end='\n')

                elif (self.prod_price()[product_name] is None) or (float(self.prod_price()[product_name]) < 0):
                    # Print error message on screen for product with negative price or no price
                    print("Invalid Product!!Cost is -ve or not found.Please select valid product.")
                    print("Below are the products and their cost per unit")
                    for k, v in self.prod_price().items():
                        # choose print to display message on the screen
                        print(k + ' $' + str(v))

                elif (cust_name not in Records.customers) and (float(self.prod_price()[product_name]) == 0):
                    # Print error message on screen for product with 0 price for a new customer
                    print("Error:New customer can not order free product.Please select Valid product")
                    print("Below are the products and their cost per unit")
                    for k, v in self.prod_price().items():
                        print(k + " :$" + str(v))
                else:
                    break

            # Using while loop until the user enters  quantity less than or equal to stock of the product
            while True:
                # Accept input of the quantity of the grocery item purchased
                quantity = int(input("quantity purchased\n"))
                # Using for loop to check user input with our pre-defined stock list
                if int(quantity) > int(self.prod_stock()[product_name]):
                    # Print error message when quantity entered is greater than stock available
                    print("Sorry!!Out of stock.Quantity enter must be less than or equal to stock")
                    # Print product list with respective stock
                    print("Below are the product and their stock")
                    for k, v in self.prod_stock().items():
                        print(k + " : " + str(v))
                else:

                    break
            item_total = 0
            # Used if-else condition to implement discount logic for existing customer
            if cust_name not in self.cust_name.values():
                item_total = item_total + (quantity * self.prod_price()[product_name])

            elif cust_name in self.special_customer and self.custinst[cust_name] == 'R':
                item_total = (quantity * self.prod_price()[product_name]) - (
                        quantity * self.prod_price()[product_name] * self.special_discount[cust_name])

            elif cust_name in self.cust_name.values() and self.custinst[cust_name] == 'R':
                item_total = (quantity * self.prod_price()[product_name]) - (
                        quantity * self.prod_price()[product_name] * RetailCustomer.rate_of_discount)

            elif (cust_name in self.special_customer) and (self.custinst[cust_name] == 'W'):
                y = (quantity * self.prod_price()[product_name])
                if y <= WholesaleCustomer.threshold:
                    item_total = y - (quantity * self.prod_price()[product_name] * self.special_discount[cust_name])
                elif y > WholesaleCustomer.threshold:
                    item_total = y - (
                            (quantity * self.prod_price()[product_name]) * (
                            float(self.special_discount[cust_name]) + float(.05)))

            elif (cust_name in self.cust_name.values()) and (self.custinst[cust_name] == 'W'):
                x = (quantity * self.prod_price()[product_name])
                if x <= self.get_threshold():
                    item_total = x - (quantity * self.prod_price()[product_name] * WholesaleCustomer.lower_discount)
                else:
                    item_total = x - (quantity * self.prod_price()[product_name] * WholesaleCustomer.upper_discount)

            for data in Records.products:
                if product_name == data["name"]:
                    data["instockQuantity"] = str(int(data["instockQuantity"]) - int(quantity))
            for data in Records.combdata:
                if product_name == data["combo_name"]:
                    data["instockQuantity"] = str(int(data["instockQuantity"]) - int(quantity))
            self.prod_stock()[product_name] = (int(self.prod_stock()[product_name]) - int(quantity))
            remaining_quantity = self.prod_stock()[product_name]

            # Create a dictionary entry which contains the name,number and price entered by user, since it is
            # easy to identify customer name ,item name , quantity and their corresponding per unit cost in a dictionary
            grocery_item = {'customer_name': cust_name, 'item_name': product_name, 'quantity': int(quantity),
                            'cost': float(self.prod_price()[product_name]), 'total_cost': float(item_total),
                            'stock_remaining': int(remaining_quantity)}

            # Add the grocery_item to the grocery_history list using the append function to maintain the order history
            # for the further analysis.
            self.grocery_history.append(grocery_item)

            # initializing group key
            grp_key = 'customer_name'

            # initializing sum keys
            sum_keys = ['total_cost']

            # Summation Grouping in Dictionary List to get total order value for all customers
            res = {}
            for sub in self.grocery_history:
                ele = sub[grp_key]
                if ele not in res:
                    res[ele] = {x: 0 for x in sum_keys}
                for y in sum_keys:
                    res[ele][y] += float(sub[y])

            for item in self.customers:
                if item['name'] == cust_name and item['customerType'] == 'R':
                    item['total'] = res[cust_name]['total_cost']
                    item['discountRate'] = RetailCustomer.rate_of_discount * 100
                elif item['name'] == cust_name and item['customerType'] == 'W':
                    if res[cust_name]['total_cost'] < WholesaleCustomer.threshold:
                        item['total'] = res[cust_name]['total_cost']
                        item['discountRate'] = WholesaleCustomer.lower_discount * 100
                    else:
                        item['total'] = res[cust_name]['total_cost']
                        item['discountRate'] = WholesaleCustomer.upper_discount * 100

            # Accept input from the user asking if they have finished entering grocery items
            response = input("would you like to enter another item?\nType 'c' for continue or 'q' to quite:\n")

            # choose if loop to end the while loop
            if response == 'q':
                stop = True

        for item in self.grocery_history:
            # Print to display message on the screen and .format method format the output
            print("{:<8}purchased {:>2} x {:>2}".format(item['customer_name'], item['item_name'], item['quantity']))
            # Print to display message on the screen.%.2f to display floating number up to two decimal place
            print("Unit Cost: $%.2f" % item['cost'])
            # Print to display message on the screen.%.2f to display floating number up to two decimal place
            print("Total Cost: $%.2f" % item['total_cost'])
            # Print to display message on the screen to display the remaining stock quantity
            print("Remaining Stock : {:>2}".format(item['stock_remaining'])+'\n')

        return self.grocery_history

    # Function to place an order from file
    def order_place_file(self):
        file_order = []
        customer_detail = []
        product_detail = []
        quantity_detail = {}
        # Ask user to enter file name
        print("Taking order from File")
        fileName = input("enter the file name:\n")
        # for handling file open error
        try:
            f = open(fileName, "r")
            data = f.read()
            lines = data.split('\n')
            for line in lines:
                items = line.split(', ')
                obj = {
                    'customer': items[0],
                    'product': items[1],
                    'quantity': items[2],
                }
                # adding new item in the list
                file_order.append(obj)
        except IOError:
            # if file not found then print err message and exit
            print("Error: Unable to open the file you entered.")
            exit()
        for item in file_order:
            customer_detail.append(item['customer'])
            product_detail.append(item['product'])
            quantity_detail.setdefault(item['product'], item['quantity'])

        for item in file_order:
            if len(item['product']) == 2:
                item_total = int(item['quantity']) * self.pd_price()[item['product']] * 0.9
                for data in Records.products:
                    if item['product'] == data["id"]:
                        data["instockQuantity"] = str(int(data["instockQuantity"]) - int(item['quantity']))
                for data in Records.combdata:
                    if item['product'] == data["comb_id"]:
                        data["instockQuantity"] = str(int(data["instockQuantity"]) - int(item['quantity']))
                self.prod_id_stock()[item['product']] = (
                        int(self.prod_id_stock()[item['product']]) - int(item['quantity']))
                remaining_quantity = self.prod_id_stock()[item['product']]

                # Create a dictionary entry which contains the name,number and price entered by user, since it is easy
                # to identify customer name ,item name , quantity and their corresponding per unit cost in a dictionary
                grocery_item = {'customer_name': item['customer'], 'item_name': item['product'],
                                'quantity': int(item['quantity']),
                                'cost': float(self.pd_price()[item['product']]), 'total_cost': float(item_total),
                                'stock_remaining': int(remaining_quantity)}
                self.grocery_history.append(grocery_item)
            else:
                item_total = int(item['quantity']) * self.prod_price()[item['product']] * 0.9
                for data in Records.products:
                    if item['product'] == data["name"]:
                        data["instockQuantity"] = str(int(data["instockQuantity"]) - int(item['quantity']))
                for data in Records.combdata:
                    if item['product'] == data["combo_name"]:
                        data["instockQuantity"] = str(int(data["instockQuantity"]) - int(item['quantity']))
                self.prod_stock()[item['product']] = (
                        int(self.prod_stock()[item['product']]) - int(item['quantity']))
                remaining_quantity = self.prod_stock()[item['product']]
                # Create a dictionary entry which contains the name,number and price entered by user, since it is easy
                # to identify customer name ,item name , quantity and their corresponding per unit cost in a dictionary
                grocery_item = {'customer_name': item['customer'], 'item_name': item['product'],
                                'quantity': int(item['quantity']),
                                'cost': float(self.prod_price()[item['product']]), 'total_cost': float(item_total),
                                'stock_remaining': int(remaining_quantity)}
                self.grocery_history.append(grocery_item)

        for item in self.grocery_history:
            # Print to display message on the screen and .format method format the output
            print("{:<8}purchased {:>2} x {:>2}".format(item['customer_name'], item['item_name'], item['quantity']))
            # Print to display message on the screen.%.2f to display floating number up to two decimal place
            print("Unit Cost: $%.2f" % item['cost'])
            # Print to display message on the screen.%.2f to display floating number up to two decimal place
            print("Total Cost: $%.2f" % item['total_cost'])
            # Print to display message on the screen to display the remaining stock quantity
            print("Remaining Stock : {:>2}".format(item['stock_remaining']))

        print("Order Placed!!Thank You")

    # Function to find most valuable customer
    def valuable_customer(self):
        # initializing group key
        grp_key = 'customer_name'

        # initializing sum keys
        sum_keys = ['total_cost']

        # Summation Grouping in Dictionary List to get total order value for all customers
        res = {}
        for sub in self.grocery_history:
            ele = sub[grp_key]
            if ele not in res:
                res[ele] = {x: 0 for x in sum_keys}
            for y in sum_keys:
                res[ele][y] += float(sub[y])

        # # Using for loop to iterate through the res dictionary
        for k, v in res.items():
            # Assigning key and value of res to order_value dictionary
            self.order_value[k] = v['total_cost']

        # Declare a variable to store customer with maximum order value present in order_value dictionary
        most_valuable = max(self.order_value, key=self.order_value.get)
        # Print most valuable customer on the screen
        print("Most valuable customer:" + str(most_valuable) + " $" + str(self.order_value[most_valuable]) + '\n')

    # Function to display order details
    def order_details(self):
        # Print to display message on the screen and .format method format the output.
        print("{:<15} {:<15} {:<15} {:<15}".format('customer_name', 'item_name', 'quantity', 'cost'))
        # creating separator
        print("-" * 60)
        # Using for loop to iterate through all elements of grocery_history
        for item in self.grocery_history:
            # Print to display message on the screen and .format method format the output
            print("{:<15} {:<15} {:<15} ${:<15}\n".format(item['customer_name'], item['item_name'], item['quantity'],
                                                          item['cost']))

    def reset_discount(self):

        while True:
            sp_cust_name = input("Please enter the customer name for special discount: \n")
            # Using if-else loop to check user input with our pre-defined product list
            if sp_cust_name not in self.cust_name.values():
                # Print error message on screen for customer not present in inventory
                print("Sorry!! You are not existing customer, only existing customers can reset discount rate")
            else:
                break
        self.special_customer.append(sp_cust_name)
        sp_cust_disc = float(input("Please enter the discount Rate (e.g. 10, 15, 20..): \n"))
        self.special_discount.setdefault(sp_cust_name, sp_cust_disc / 100)

# Function for taking user choice from the menu
def takeUserChoice():
    try:
        print("Select an option:")

        print("1). Search a customer:")

        print("2). Search a product:")

        print("3). List all customers:")

        print("4). List all products:")

        print("5). Place an Order:")

        print("6). Place an Order from file:")

        print("7). Replenish:")

        print("8). Most valuable customer: ")

        print("9). Order details: ")

        print("10). Update threshold for WholesaleCustomer: ")

        print("11). Reset discount rate for a customer: ")

        print("0). Exit.")

        choice = int(input("Enter your choice: "))

        return choice
    # Except block to handle error if user enters invalid choice
    except ValueError:
        print("Invalid choice.Please choose correct option from the menu")


# Required main function

def main():
    rec = Records()
    prod = Product()
    comb = combo()
    ord = Order()
    cust = WholesaleCustomer()

    # reading both the files
    rec.readCustomers()

    rec.readProducts()

    prod.prod_name()
    prod.prod_price()
    prod.prod_stock()
    prod.prod_id()
    prod.prd_id_price()
    prod.prod_id_stock()
    prod.pd_price()

    comb.getprice()

    # Keeping user in the loop until he/she select to exit

    while (1):

        choice = takeUserChoice()

        # Call function as per customer selection from menu options

        if choice == 1:

            val = input("Enter customer's name or id: ")

            customer = rec.findCustomer(val)

            if customer != None:

                print("\nCustomer information:")

                print("\tID: " + customer['id'])

                print("\tName: " + customer['name'])

                print("\tCustomer type: " + customer['customerType'])

                print("\tDiscount rate: " + customer['discountRate'])

                print("\tTotal: " + customer['total'] + '\n')

            else:

                print("\nCustomer not found\n")

        elif choice == 2:

            val = input("Enter product's  id: ")

            rec.findProduct(val)


        elif choice == 3:

            rec.listCustomers()

        elif choice == 4:

            rec.listProducts()

        elif choice == 5:

            ord.usercreate()

        elif choice == 6:

            ord.order_place_file()

        elif choice == 7:

            prod.replenish()

        elif choice == 8:

            ord.valuable_customer()

        elif choice == 9:

            ord.order_details()

        elif choice == 10:

            value = int(input("Enter value of threshold: "))

            cust.set_threshold(value)
            x = cust.get_threshold()
            print("updated Threshold for wholesale customer: "+ str(x) + '\n')

        elif choice == 11:

            ord.reset_discount()

        elif choice == 0:

            print("Thank you!")

            break

        else:

            print("Please enter a valid choice.\n")


main()
