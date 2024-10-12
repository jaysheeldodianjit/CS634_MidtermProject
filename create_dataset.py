import csv
import random

general_store_items = ["Milk", "Bread", "Diapers", "Eggs", "Butter", "Chips", "Soda", "Apples", "Chicken", "Toilet Paper"]
pet_store_items = ["Dog Food", "Cat Litter", "Chew Toys", "Pet Shampoo", "Fish Tank", "Bird Seed", "Leash", "Scratching Post", "Pet Bed", "Flea Collar"]
electronic_store_items = ["Smartphone", "Laptop", "Headphones", "Smart TV", "Tablet", "Smartwatch", "Gaming Console", "Wireless Mouse", "External Hard Drive", "Bluetooth Speaker"]
sportswear_store_items = ["Running Shoes", "Athletic Shorts", "Sports Bra", "Compression Socks", "Moisture-Wicking T-shirt", "Yoga Pants", "Sweatband", "Tracksuit", "Gym Bag", "Water Bottle"]
gardening_items = [
    "Flower Seeds", "Vegetable Seeds", "Fertilizer", 
    "Gardening Gloves", "Shovel", "Rake", 
    "Watering Can", "Potting Soil", "Planter", "Garden Hose"
]

general_store_rules = [
    ("Milk", "Bread"),
    ("Diapers", "Eggs"),
    ("Chips", "Soda"),
    ("Milk", "Bread", "Butter"),
    ("Chips", "Soda", "Apples"),
    ("Milk", "Bread", "Eggs", "Butter")
]

pet_store_rules = [
    ("Dog Food", "Chew Toys"),
    ("Cat Litter", "Scratching Post"),
    ("Fish Tank", "Pet Shampoo"),
    ("Dog Food", "Leash", "Pet Bed"),
    ("Cat Litter", "Scratching Post", "Flea Collar"),
    ("Dog Food", "Chew Toys", "Leash", "Pet Shampoo")
]

electronic_store_rules = [
    ("Smartphone", "Headphones"),
    ("Laptop", "Wireless Mouse"),
    ("Gaming Console", "Smart TV"),
    ("Smartphone", "Smartwatch", "Headphones"),
    ("Laptop", "External Hard Drive", "Wireless Mouse"),
    ("Smartphone", "Laptop", "Tablet", "External Hard Drive")
]

sportswear_store_rules = [
    ("Running Shoes", "Athletic Shorts"),
    ("Sports Bra", "Moisture-Wicking T-shirt"),
    ("Yoga Pants", "Water Bottle"),
    ("Running Shoes", "Compression Socks", "Moisture-Wicking T-shirt"),
    ("Sports Bra", "Yoga Pants", "Sweatband"),
    ("Running Shoes", "Athletic Shorts", "Moisture-Wicking T-shirt", "Gym Bag")
]


# Rules for the gardening supplies
gardening_rules = [
    ("Flower Seeds", "Fertilizer"),
    ("Vegetable Seeds", "Potting Soil"),
    ("Gardening Gloves", "Shovel"),
    ("Watering Can", "Garden Hose"),
    ("Rake", "Shovel", "Fertilizer"),
    ("Planter", "Flower Seeds"),
    ("Vegetable Seeds", "Watering Can"),
    ("Fertilizer", "Potting Soil", "Garden Hose"),
    ("Flower Seeds", "Gardening Gloves"),
    ("Planter", "Vegetable Seeds", "Watering Can")
]


MIN_ITEMS = 2
MAX_ITEMS = 4

def generate_transaction(rules, items, min_items, max_items):
    transaction = set()

    # Apply rules deterministically with a 70% chance
    for rule in rules:
        if random.random() < 0.7:
            transaction.update(rule)

    while len(transaction) < min_items:
        random_item = random.choice(items)
        transaction.add(random_item)

    if len(transaction) > max_items:
        transaction = set(random.sample(list(transaction), max_items))

    return list(transaction)

def create_dataset(num_transactions, rules, items, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["TID", "ITEM_SET"])

        for tid in range(1, num_transactions + 1):
            transaction = generate_transaction(rules, items, MIN_ITEMS, MAX_ITEMS)
            writer.writerow([tid, ','.join(transaction)])

create_dataset(200, general_store_rules, general_store_items, 'dataset/general_store_dataset.csv')
create_dataset(200, pet_store_rules, pet_store_items, 'dataset/pet_store_dataset.csv')
create_dataset(200, electronic_store_rules, electronic_store_items, 'dataset/electronic_store_dataset.csv')
create_dataset(200, sportswear_store_rules, sportswear_store_items, 'dataset/sportswear_store_dataset.csv')
create_dataset(200, gardening_rules, gardening_items, 'dataset/gardening_dataset.csv')

print("All datasets have been generated successfully.")