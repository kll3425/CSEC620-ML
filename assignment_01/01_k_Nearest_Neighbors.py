import numpy as np
from collections import Counter

'''
tf-idf functions taken from the tokenizer file
'''  
def term_frequency(word, document):
    # Count the frequency of the word in the document (list of words)
    return document.count(word) / len(document)

def document_frequency(word, documents):
    # Count the number of documents containing the word
    count = 0
    for document in documents:
        if word in document:
            count += 1
    return count

#idf and tf_idf need modification. i'm thinking 1 document variable
def idf(word, documents):
    # Calculate the inverse document frequency of the word
    df = document_frequency(word, documents)
    # If the word is not in any document, return 0
    # This prevents division by zero errors when calculating idf
    if df == 0:
        return 0
    return np.log(len(documents) / (document_frequency(word, documents)))

def tf_idf(word, document, documents):
    # Calculate the term frequency-inverse document frequency of the word
    return term_frequency(word, document) * idf(word, documents)

'''
Copilot Prompt: Can you code me a function to calculate 
Euclidean distance?

Example Usage:
point1 = [1, 2, 3]
point2 = [4, 5, 6]
distance = euclidean_distance(point1, point2)
print(f"Euclidean Distance: {distance}")
'''
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2)) 

'''
Copilot prompt:
Please generate code to predict k-nearest neighbors

import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def knn_predict(training_data, training_labels, test_point, k):
    distances = []
    for i in range(len(training_data)):
        dist = euclidean_distance(test_point, training_data[i])
        distances.append((dist, training_labels[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for _, label in distances[:k]]
    return Counter(k_nearest_labels).most_common(1)[0][0]

def main():
    # Example training data and labels
    training_data = [
        [1, 2],
        [2, 3],
        [3, 4],
        [6, 7],
        [7, 8],
        [8, 9]
    ]
    training_labels = ['A', 'A', 'A', 'B', 'B', 'B']
    
    # Example test point
    test_point = [5, 5]
    
    # Number of neighbors
    k = 3
    
    # Predict the label of the test point
    prediction = knn_predict(training_data, training_labels, test_point, k)
    print(f"The predicted label for the test point {test_point} is: {prediction}")

if __name__ == "__main__":
    main()
'''

def main():
    '''
    Separate each term into its own item in a list
    '''
    with open("SMSSpamCollection") as file:
        for line in file:
            line = line.strip().split()
            for word in line:
                term_frequency(word, file)
                document_frequency(word, file)            
    
    
            
if __name__ == "__main__":
    main()