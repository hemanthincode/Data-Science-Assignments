# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 08:58:00 2024

@author: bommi
"""



import pandas as pd
from apyori import apriori
import matplotlib.pyplot as plt

# Step 1: Read the book dataset
df = pd.read_csv('book.csv', encoding='latin') 

# Step 2: Prepare data for Apriori
transactions = []
for i in range(len(df)):
    transactions.append([str(df.values[i, j]) for j in range(len(df.columns))])

# Step 3: Try different values of support and confidence
support_values = [0.1, 0.2, 0.3]
confidence_values = [0.5, 0.6, 0.7]

# Store results for each combination of support and confidence
results_list = []

for support in support_values:
    for confidence in confidence_values:
        print(f"\nSupport: {support}, Confidence: {confidence}")
        rules = apriori(transactions=transactions, 
                        min_support=support, 
                        min_confidence=confidence, 
                        min_length=2)
results = list(rules)
results_list.append({'Support': support, 'Confidence': confidence, 'Rules': len(results)})

# Extract values for scatter plot from the best combination of support and confidence
best_result = max(results_list, key=lambda x: x['Rules'])
best_support = best_result['Support']
best_confidence = best_result['Confidence']

# Step 5: Visualize obtained rules using a scatter plot
rules = apriori(transactions=transactions, 
                min_support=best_support, 
                min_confidence=best_confidence, 
                min_length=2)
results = list(rules)

# Scatter plot of Support vs Confidence with color gradient based on Lift
lift_values = [rule.ordered_statistics[0].lift for rule in results]

plt.scatter([best_support], [best_confidence], c='red', marker='*', s=200, label='Best Combination')
plt.scatter(support_values, confidence_values, c=lift_values, cmap='viridis', alpha=0.5)
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules Scatter Plot with Lift')
plt.legend()
plt.show()
