# Import the itertools and pandas libraries
from itertools import combinations
import pandas as pd


# Function to generate all possible combinations of items in a transaction
def generateCombinations(items):
    # Generate all possible combinations of items
    combinations_list = []
    for i in range(1, len(items) + 1):
        combinations_list.extend(list(combinations(items, i)))
    return combinations_list


# Function to check if an itemset is frequent
def checkFrequentItems(itemset, transactions, min_support):
    # Count the number of occurrences of the itemset in the transactions
    count = 0
    for transaction in transactions:
        if set(itemset).issubset(set(transaction)):
            count += 1
    # Check if the itemset meets the minimum support threshold
    if count / len(transactions) >= min_support:
        return True, count
    else:
        return False, count


# Function to generate association rules
def generateAssociationRules(frequent_itemsets, transactions, min_confidence):
    # Generate all possible rules from frequent itemsets
    rules = []
    for itemset in frequent_itemsets:
        itemset = list(itemset)
        if len(itemset) > 1:
            for i in range(len(itemset)):
                A = itemset[:i] + itemset[i + 1:]
                B = [itemset[i]]
                confidence = getConfidence(A, B, transactions)
                if confidence >= min_confidence:
                    rules.append((A, B, confidence))
    return rules


# Function to calculate the confidence of a rule
def getConfidence(A, B, transactions):
    # Calculate the support of the A and the rule
    A_count = 0
    union = 0
    for transaction in transactions:
        if set(A).issubset(set(transaction)):
            A_count += 1
            if set(B).issubset(set(transaction)):
                union += 1
    # Calculate the confidence
    if A_count == 0:
        return 0
    else:
        return union / A_count


# Read the data from the Excel file
df = pd.read_excel("CoffeeShopTransactions.xlsx")

# Extract the transaction data from the DataFrame
df = df.iloc[:, 3:]
transactions = []
for _, row in df.iterrows():
    transactions.append([item.lower().replace(" ", "") for item in row if item == 'Caramel bites' or 'CaramelBites'])


# Set the minimum support threshold
min_support = float(input('Enter the minimum support from 0 to 1: '))

# Generate frequent itemsets using the Apriori algorithm
frequent_itemsets = []
sup_count = []
items = set([item for transaction in transactions for item in transaction])
previous_frequent_itemsets = []
for i in range(1, len(items) + 1):
    for combination in combinations(items, i):
        is_frequent, count = checkFrequentItems(combination, transactions, min_support)
        if is_frequent:
            if count >= min_support:
                frequent_itemsets.append(combination)
                sup_count.append(count)
                previous_frequent_itemsets.append(combination)

    items = set([item for frequent_itemset in previous_frequent_itemsets for item in frequent_itemset])
    previous_frequent_itemsets = []

# Set the minimum confidence threshold
min_confidence = float(input('Enter the minimum confidence from 0 to 1: '))

# Generate association rules from the frequent itemsets
rules = generateAssociationRules(frequent_itemsets, transactions, min_confidence)

# Print the results
print("Frequent Itemsets:")
for itemset, count in zip(frequent_itemsets, sup_count):
    print(itemset, count)
print("\nAssociation Rules:")
for rule in rules:
    print("{} -> {}: {}".format(rule[0], rule[1], rule[2]))
