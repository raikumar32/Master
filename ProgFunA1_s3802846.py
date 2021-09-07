# Declare list to store the customer name
customer = ['viper', 'sage', 'cypher', 'sova']
# Declare dictionary to store list of products so that product identifier and product name can be displayed.
product = {'product1': 'p1', 'product2': 'p2', 'product3': 'p3', 'product4': 'p4', 'product5': 'p5'}
# Declare dictionary to store stock of product since it is easy to map products with their corresponding stock
stock = {'p1': 30, 'p2': 40, 'p3': 20, 'p4': 10, 'p5': 5}
# Declare dictionary to store cost per unit of product since it is easy to map product with their corresponding cost
cost = {'p1': 2, 'p2': 3, 'p3': -1, 'p4': 0, 'p5': None}
# Declare dictionary to store the customer name, quantity, product name and cost per unit to map their value with key
grocery_item = {}
# Declare multi-dimensional array to store all order history so that it can be displayed in tabular form
grocery_history = []
# Declare dictionary to store the customer name to get most valuable customer customer
order_value = {}


# Created function so that different functionality can be implemented separately
def mainmenu():
    # Used while loop to create an infinite loop until a condition is satisfied
    while True:
        # Display menu options on the screen
        print('===============================')
        print('= Inventory Management System =')
        print('===============================')
        print('[1] Display all customers: ')
        print('[2] Display products and their corresponding cost per unit: ')
        print('[3] Place an Order: ')
        print('[4] Replenish: ')
        print('[5] Most valuable customer: ')
        print('[6] Order details: ')
        print('[7] Replace existing product and price list: ')
        print('[8] Quit: ')
        # Taking input from user for menu selection
        choice = int(input("Enter choice: "))
        # Validate user selection from the user
        try:
            # Used if elsif and else to call respective function block based on user choice
            if choice == 1:
                display_customer()
            elif choice == 2:
                display_product_price()
            if choice == 3:
                order_place()
            elif choice == 4:
                replenish()
            elif choice == 5:
                valuable_customer()
            elif choice == 6:
                order_details()
            elif choice == 7:
                rpl_product_cost()
            else:
                exit()
        # Except block to handle error if user enters invalid choice
        except ValueError:
            print("Invalid choice.Enter 1-8")


# Function for menuoption 3
def order_place():
    # Calling global variables
    global customer
    global product
    global stock
    global order_value
    global cost
    global grocery_item
    global grocery_history

    # variable used to check if the while loop condition is met
    stop = False

    # Accept user entered customer name
    cust_name = input("Enter your name \n")

    # Using while loop to take customer order until he/she is finished with his/her order
    while not stop:

        # Using while loop until user enters a valid product
        while True:
            # Accept input of the name of the grocery item purchased
            prod_name = input("Product name \n")
            # Using if-else loop to check user input with our pre-defined product list
            if prod_name not in product.values():
                # Print error message on screen for product not present in inventory
                print("Invalid Product!!Please enter product present in our inventory")
                # Print products available in inventory
                print("Below are the products available")
                for k, v in product.items():
                    print(k + " : " + v)
            elif (cost[prod_name] is None) or (cost[prod_name] < 0):
                # Print error message on screen for product with negative price or no price
                print("Invalid Product!!Cost is not found.Please select valid product.")
                print("Below are the products and their cost per unit")
                for k, v in cost.items():
                    # choose print to display message on the screen
                    print(k + " :$" + str(v))
            elif (cust_name not in customer) and (cost[prod_name] == 0):
                # Print error message on screen for product with 0 price for a new customer
                print("Error:New customer can not order free product.Please select Valid product")
                print("Below are the products and their cost per unit")
                for k, v in cost.items():
                    print(k + " :$" + str(v))
            else:
                break

        # Using while loop until the user enters  quantity less than or equal to stock of the product
        while True:
            # Accept input of the quantity of the grocery item purchased
            quantity = int(input("quantity purchased\n"))
            # Using for loop to check user input with our pre-defined stock list
            if int(quantity) > stock[prod_name]:
                # Print error message when quantity entered is greater than stock available
                print("Sorry!!Out of stock.Quantity enter must be less than or equal to stock")
                # Print product list with respective stock
                print("Below are the product and their stock")
                for k, v in stock.items():
                    print(k + " : " + str(v))
            else:
                break
        # Used if-else condition to implement discount logic for existing customer
        if cust_name in customer:
            item_total = quantity * cost[prod_name] * 0.9
        else:
            item_total = quantity * cost[prod_name]

        # Create a dictionary entry which contains the name,number and price entered by user, since it is
        # easy to identify customer name ,item name , quantity and their corresponding per unit cost in a dictionary
        grocery_item = {'customer_name': cust_name, 'item_name': prod_name, 'quantity': int(quantity),
                        'cost': cost[prod_name], 'total_cost': float(item_total)}

        # Add the grocery_item to the grocery_history list using the append function to maintain the order history
        # for the further analysis.
        grocery_history.append(grocery_item)

        # Accept input from the user asking if they have finished entering grocery items
        response = input("would you like to enter another item?\nType 'c' for continue or 'q' to quite:\n")

        # choose if loop to end the while loop
        if response == 'q':
            stop = True

    # initializing group key
    grp_key = 'customer_name'

    # initializing sum keys
    sum_keys = ['total_cost']

    # Summation Grouping in Dictionary List to get total order value for all customers
    res = {}
    for sub in grocery_history:
        ele = sub[grp_key]
        if ele not in res:
            res[ele] = {x: 0 for x in sum_keys}
        for y in sum_keys:
            res[ele][y] += float(sub[y])
    # print("The grouped list : " + str(res))

    for item in grocery_history:
        # Print to display message on the screen and .format method format the output
        print("{:<8}purchased {:>2} x {:>2}".format(item['customer_name'], item['item_name'], item['quantity']))
        # Print to display message on the screen.%.2f to display floating number up to two decimal place
        print("Unit Cost: $%.2f" % item['cost'])
        # Print to display message on the screen.%.2f to display floating number up to two decimal place
        print("Total Cost: $%.2f" % item['total_cost'])

    # Using for loop to iterate through the res dictionary
    for k, v in res.items():
        # Assigning key and value of res to order_value dictionary
        order_value[k] = v['total_cost']
    # choose if statement to check if user entered customer is present in our pre-defined customer list ,if not
    # then appending it to customer list using .append method
    if cust_name not in customer:
        customer.append(cust_name)

    # Returning to mainmenu
    mainmenu()


# Function for menuoption 4
def replenish():
    # Calling global variable
    global stock
    # Define a variable to take user input so that it can be used to replace stock of all products with user entered
    # value.
    x = int(input("Enter quantity to replenish the stock: \n"))
    # Using for loop  and .items() function to iterate through all key and value of stock dictionary.
    for k, v in stock.items():
        # Updating product stock with user entered value
        stock[k] = x
    print("Stock Updated!!")
    for k, v in stock.items():
        print(k + " : " + str(v))

    # Returning to mainmenu
    mainmenu()


# Function for menuoption 6
def order_details():
    # Calling global variable
    global grocery_history
    # Print to display message on the screen and .format method format the output.
    print("{:<15} {:<15} {:<15} {:<15}".format('customer_name', 'item_name', 'quantity', 'cost'))
    # creating separator
    print("-" * 60)
    # Using for loop to iterate through all elements of grocery_history
    for item in grocery_history:
        # Print to display message on the screen and .format method format the output
        print("{:<15} {:<15} {:<15} ${:<15}\n".format(item['customer_name'], item['item_name'], item['quantity'],
                                                      item['cost']))
    # Returning to mainmenu
    mainmenu()


# Function for menuoption 5
def valuable_customer():
    # Calling global variable
    global order_value
    # Declare a variable to store customer with maximum order value present in order_value dictionary
    most_valuable = max(order_value, key=order_value.get)
    # Print most valuable customer on the screen
    print("Most valuable customer:" + str(most_valuable) + " $" + str(order_value[most_valuable]))

    # Returning to mainmenu
    mainmenu()


# Function for menuoption 7
def rpl_product_cost():
    # Calling global variables
    global product
    global cost
    global stock

    # Defined a list to store comma separated user input.
    # Using for loop through the user input and splitting it based on "," and  storing it in defined list.
    new_product_list = [str(item) for item in input(" Enter the comma separated product list: \n").split(",")]
    # Using zip function to returns zip object with new user input value and assigning it back to product dictionary
    product = dict(zip(list(product.keys()), new_product_list))
    # Using zip function to returns zip object with new user input value and assigning it back to stock dictionary
    stock = dict(zip(new_product_list, list(stock.values())))
    # Using zip function to return zip object with new user input value and assigning it back to cost dictionary
    cost = dict(zip(new_product_list, list(cost.values())))

    # Using for loop to iterate through key and value of dictionary Product and then printing it to screen.
    for k, v in product.items():
        # choose print to display message on the screen
        print(k + " : " + str(v))

    # Defined a list to store comma separated user input.
    # Using for loop through the user input and splitting it based on "," and  storing it in defined list.
    new_cost_list = [float(item) for item in input(" Enter the comma separated cost of the product: \n").split(",")]
    # Using zip function to return zip object with new user input value and assigning it back to cost dictionary
    cost = dict(zip(list(cost.keys()), new_cost_list))
    # Using for loop to iterate through values of product
    for v in product.values():
        # Using if loop condition to validate whether value of product which is a key for cost is present or not.
        # If not, setting it to None value
        if v not in cost.keys():
            # setting product price to none value for which user did not enter the price
            cost[v] = None
    # Using for loop to iterate through key and value of dictionary.
    for k, v in cost.items():
        # Display message on the screen
        print(k + " :$" + str(v))

    # Returning to mainmenu
    mainmenu()


# Function for menuoption 1
def display_customer():
    # Calling global variable
    global customer
    # Using join() function to join a sequence of strings of customer list in its arguments with the string
    # Display customer list on screen
    print("Below is the list of customers: \n" + ' '.join(customer))

    # Returning to mainmenu
    mainmenu()


# Function for menuoption 2
def display_product_price():
    # Calling global variable
    global cost
    # Display product and cost list on screen
    print("Below is the list of products and their corresponding cost per unit ")
    # Using for loop to iterate through key and value of dictionary and then printing it to screen
    for k, v in cost.items():
        print(k + " :$" + str(v))

    # Returning to mainmenu
    mainmenu()


mainmenu()
